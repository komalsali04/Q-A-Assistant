# Q&A Assistant - PDF Document Question Answering System

A powerful RAG (Retrieval-Augmented Generation) application that allows you to upload PDF documents and ask questions about their content. Built with LangChain, ChromaDB, HuggingFace embeddings, and Ollama LLM.

## Features

-  **PDF Document Processing** - Upload and process any PDF document
-  **Intelligent Retrieval** - Uses similarity search to find relevant context
-  **Interactive Chat Interface** - Clean Streamlit UI for asking questions
-  **Fast Embeddings** - Optimized HuggingFace embeddings for quick processing
-  **Smart Caching** - Automatic database caching (same PDF won't be reprocessed)
-  **Contextual Answers** - Answers based solely on your document content

## Prerequisites

Before installing, ensure you have the following:

1. **Python 3.11+** installed on your system
2. **Ollama** installed and running ([Download Ollama](https://ollama.ai))

### Pull Required Ollama Model

You need to pull the Mistral model for the LLM:

```bash
ollama pull mistral
```

This will download the Mistral model (approximately ~4.1 GB) which will be used for generating answers.

## Installation

1. **Clone or download this repository**

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**

   **On Windows:**
   ```bash
   venv\Scripts\activate
   ```

   **On macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   This will install all required packages including LangChain, ChromaDB, Streamlit, and other dependencies.

5. **Verify Ollama is running**

   ```bash
   ollama list
   ```

   You should see `mistral` in the list. If not, run `ollama pull mistral` again.

## Usage

### Running the Streamlit Web Application (Recommended)

To launch the interactive web UI:

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`.

**Steps to use:**
1. Click on the sidebar and upload a PDF file
2. Wait for the PDF to be processed (first time only - subsequent uploads of the same file are cached)
3. Type your question in the chat interface
4. Receive contextual answers based on your document

### Running the Command-Line Script

Alternatively, you can use the command-line version:

```bash
python bot.py
```

**Note:** For the command-line version, you'll need to modify `bot.py` to specify your PDF file path in the `load_documents()` function.

## Architecture

### Chunking Strategy

The application uses the following chunking parameters:

- **Chunk Size: 1000 characters** - This size balances semantic coherence with retrieval granularity. It's large enough to preserve context and meaning within chunks, while small enough to enable precise retrieval of relevant information.
  
- **Chunk Overlap: 150 characters** - The overlap ensures continuity between adjacent chunks, preventing important information from being split across chunk boundaries. This is particularly important for maintaining context when concepts span multiple sentences or paragraphs.

- **Separators: `["\n\n", "\n", ". ", " ", ""]`** - The recursive splitting prioritizes paragraph breaks first, then sentence breaks, then word boundaries. This strategy preserves document structure and keeps related content together, which improves both retrieval accuracy and answer quality.

This configuration is optimized for research papers, technical documents, and general PDF content, providing a good balance between retrieval precision and semantic completeness.

### Technical Stack

- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` - Fast, lightweight, and effective for semantic search
- **Vector Database:** ChromaDB - Persistent vector storage with automatic indexing
- **LLM:** Ollama with Mistral model - Fast CPU inference with good quality responses
- **Retrieval:** Similarity search with k=3 - Returns top 3 most relevant chunks for context
- **Framework:** LangChain - Provides the RAG pipeline orchestration

## File Structure

```
QA Assistant/
├── streamlit_app.py      # Streamlit web application
├── bot.py                # Command-line script
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── chroma_db_*/         # Generated vector databases (one per PDF)
```

## Troubleshooting

### Common Issues

1. **"Connection refused" or Ollama errors**
   - Ensure Ollama is running: `ollama serve`
   - Verify the model is pulled: `ollama list`

2. **Slow response times**
   - This is normal on CPU - LLM inference can take 10-60 seconds
   - Consider using a GPU if available (Ollama will use it automatically)

3. **Import errors**
   - Make sure your virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt --upgrade`

4. **PDF processing errors**
   - Ensure the PDF is not password-protected
   - Try a different PDF file to verify the system works

## License

This project is provided as-is for educational and personal use.

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [Ollama](https://ollama.ai/)
- UI framework: [Streamlit](https://streamlit.io/)
