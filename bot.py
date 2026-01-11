from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os

# Step3: Use an embedding model to embed the docs
# Using HuggingFace embeddings: Faster and better quality than Ollama embeddings
# Option 1 (Fastest): "sentence-transformers/all-MiniLM-L6-v2" - Very fast, good quality
# Option 2 (Best Quality): "sentence-transformers/all-mpnet-base-v2" - Best accuracy, still fast

# Step1: Load the pdf loader (PyMuPDF) to load and process the pdf docs
# PyMuPDF is more robust than PyPDF and handles complex PDFs better
def load_documents(file_path: str):
 loader = PyMuPDFLoader(file_path)
 return loader.load()

# Step2: Chunk the pdf to store in vector DB, use any text splitter
# Optimized for research papers: slightly smaller chunks for better semantic coherence
def split_documents(documents: list[Document]):
 text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000,
  chunk_overlap=150,
  length_function=len,
  separators=["\n\n", "\n", ". ", " ", ""])  
 return text_splitter.split_documents(documents)

# Step3: Use an embedding model to embed the docs
# Using HuggingFace embeddings: Faster and better quality than Ollama embeddings
# Option 1 (Fastest): "sentence-transformers/all-MiniLM-L6-v2" - Very fast, good quality
# Option 2 (Best Quality): "sentence-transformers/all-mpnet-base-v2" - Best accuracy, still fast
def embed_docs():
 embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2",
  model_kwargs={"device": "cpu"},  # Use "cuda" if GPU available
  encode_kwargs={"normalize_embeddings": True}  # Better for similarity search
 )
 return embeddings 

embed = embed_docs()
# Step4: Create a vector DB using chromaDB
def build_vectorstore(chunk: list[Document], db_path: str):
 last_page_id = None
 current_chunk_index = 0
 chunk_ids = []
 print(f"Processing {len(chunk)} chunks...")

 for ch in chunk:
  source = ch.metadata.get("source")
  page = ch.metadata.get("page")
  current_page_id = f"{source}:{page}"

  if current_page_id == last_page_id:
   current_chunk_index += 1
  else:
   current_chunk_index = 0
   last_page_id = current_page_id

  chunk_id = f"{current_page_id}:{current_chunk_index}"
  chunk_ids.append(chunk_id)

  ch.metadata["id"] = chunk_id

 print("Creating ChromaDB and adding documents...")
 db = Chroma(embedding_function=embed, persist_directory=db_path)
 db.add_documents(chunk, ids = chunk_ids)
 # Note: persist() is no longer needed - ChromaDB automatically persists when using persist_directory
 all_docs = db.get()
 print(f"Total_chunks: {len(all_docs['ids'])}\n")
 print(f"ID: {all_docs['ids'][:5]}\n")
 print(f"Metadata: {all_docs['metadatas'][:5]}")
 return db

# Load existing database or create a new one
db_path = "./chroma_langchain_db"
if os.path.exists(db_path) and os.path.exists(os.path.join(db_path, "chroma.sqlite3")):
 print("Loading existing database...")
 vector_store = Chroma(embedding_function=embed, persist_directory=db_path)
 all_docs = vector_store.get()
 print(f"Total_chunks: {len(all_docs['ids'])}\n")
 print(f"ID: {all_docs['ids'][:5]}\n")
 print(f"Metadata: {all_docs['metadatas'][:5]}")
else:
 print("Database not found. Building new database...")
 docs = load_documents()
 print(docs[0])
 chunk = split_documents(docs)
 print(chunk[0])
 vector_store = build_vectorstore(chunk, db_path)

# Step5: Create Chat template and write the chaining logic

print("\n Setting up RAG Chain...")
# Shorter prompt for faster processing - reduces token count
prompt = ChatPromptTemplate.from_template(
 """Answer the question using ONLY the provided context. Be accurate and concise.

Context:
{context}

Question: {input}

Answer:"""
)

print("Initializing LLM...")
# Optimized for CPU: limit response length, use fewer threads for smaller models
# num_predict limits max response tokens (faster generation)
# num_thread: set to number of physical CPU cores for optimal performance
llm = ChatOllama(
 model="mistral",
 temperature=0,
 num_predict=512,  # Limit response length for faster generation
 num_thread=4,  # Adjust based on your CPU cores (typically 4-8 for modern CPUs)
)

print("Creating retriever with similarity search...")
# Optimized for speed: Reduced chunks (k=3) to minimize context size
# Similarity search is faster than MMR (no fetch_k needed)
# Smaller context = faster LLM processing on CPU
retriever = vector_store.as_retriever(
 search_type="similarity",
 search_kwargs={
  "k": 3  # Reduced from 5 to 3 - less context to process
 }
)

print("Building RAG Chain...")
rag_chain = ({
 "context": RunnableLambda(lambda x: x["input"]) | retriever,
 "input": RunnablePassthrough()

}
 | prompt
 | llm
 | StrOutputParser()
)

print("Querying the model...")
question = "Why did they use RNN and ANN and Decision Trees?"
answer = rag_chain.invoke({"input": question})
print(f"\nAnswer: {answer}\n")