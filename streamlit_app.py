import streamlit as st
import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Page config
st.set_page_config(
    page_title="Q&A Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_embeddings():
    """Initialize and cache embeddings"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def load_documents(file_path: str):
    """Load PDF documents"""
    loader = PyMuPDFLoader(file_path)
    return loader.load()

def split_documents(documents: list[Document]):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_documents(documents)

def build_vectorstore(chunk: list[Document], db_path: str, embed):
    """Build vector store from chunks"""
    last_page_id = None
    current_chunk_index = 0
    chunk_ids = []

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

    db = Chroma(embedding_function=embed, persist_directory=db_path)
    db.add_documents(chunk, ids=chunk_ids)
    return db

def get_or_create_vectorstore(uploaded_file, embed):
    """Get existing vectorstore or create new one from uploaded PDF"""
    # Create a unique db path based on file name
    file_name = Path(uploaded_file.name).stem
    db_path = f"./chroma_db_{file_name}"
    
    # Check if database exists
    if os.path.exists(db_path) and os.path.exists(os.path.join(db_path, "chroma.sqlite3")):
        return Chroma(embedding_function=embed, persist_directory=db_path)
    
    # Create temporary file to save uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load and process documents
        docs = load_documents(tmp_path)
        chunks = split_documents(docs)
        vector_store = build_vectorstore(chunks, db_path, embed)
        return vector_store
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def create_rag_chain(vector_store, embed):
    """Create RAG chain"""
    prompt = ChatPromptTemplate.from_template(
        """Answer the question using ONLY the provided context. Be accurate and concise.

Context:
{context}

Question: {input}

Answer:"""
    )
    
    llm = ChatOllama(
        model="mistral",
        temperature=0,
        num_predict=512,
        num_thread=4,
    )
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    rag_chain = ({
        "context": RunnableLambda(lambda x: x["input"]) | retriever,
        "input": RunnablePassthrough()
    } | prompt | llm | StrOutputParser())
    
    return rag_chain

# Main app
st.markdown('<p class="main-header"> Q&A Assistant</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for PDF upload
with st.sidebar:
    st.header(" Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to ask questions about"
    )
    
    if uploaded_file is not None:
        st.success(f" {uploaded_file.name} uploaded")
        st.info("Processing PDF... This may take a moment.")
    else:
        st.info(" Please upload a PDF file to get started")

# Main content area
if uploaded_file is not None:
    # Initialize embeddings
    with st.spinner("Initializing embeddings..."):
        embed = get_embeddings()
    
    # Get or create vectorstore
    with st.spinner("Processing PDF and building vector database..."):
        try:
            vector_store = get_or_create_vectorstore(uploaded_file, embed)
            st.success(" PDF processed successfully!")
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.stop()
    
    # Create RAG chain
    with st.spinner("Setting up Q&A system..."):
        rag_chain = create_rag_chain(vector_store, embed)
    
    st.markdown("---")
    st.header(" Ask Questions")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = rag_chain.invoke({"input": prompt})
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.button(" Clear Chat"):
        st.session_state.messages = []
        st.rerun()

else:
    # Welcome message when no PDF is uploaded
    st.info(" Please upload a PDF file from the sidebar to start asking questions")
    
    st.markdown("""
    ### How to use:
    1. **Upload a PDF** - Use the sidebar to upload your PDF document
    2. **Wait for Processing** - The system will process your PDF (first time only)
    3. **Ask Questions** - Type your questions in the chat interface
    4. **Get Answers** - Receive answers based on the content of your PDF
    
    ### Features:
    -  Support for any PDF document
    -  Fast retrieval with optimized embeddings
    -  Contextual answers from your document
    -  Automatic caching (same PDF won't be reprocessed)
    """)
