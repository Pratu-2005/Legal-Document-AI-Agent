# Legal Document AI Agent

A Streamlit-based legal document assistant that lets users upload a PDF, process it into searchable chunks, and ask questions about the document using a Retrieval-Augmented Generation (RAG) workflow powered by Google Gemini embeddings and LLMs.

## Project Overview

This project is designed for:

- Uploading a legal document in PDF format
- Splitting the document into smaller text chunks
- Creating a FAISS vector database for semantic retrieval
- Asking natural-language questions about the document
- Returning answers grounded in the uploaded document content

The app is built for research, document review, and educational use. It is not a substitute for legal advice and should not be used as a legal decision-making tool.

## Features

- PDF upload through a Streamlit interface
- Automatic document processing and chunking
- FAISS vector store creation and local persistence
- Semantic retrieval using embeddings
- Question answering with Gemini-based language model
- Page-aware answer generation when available

## Tech Stack

- Python
- Streamlit
- LangChain
- LangChain Community
- Google Generative AI
- FAISS
- PyPDF
- Sentence Transformers

## Project Structure

- `app.py` – main Streamlit application
- `read_pdf.py` – PDF reading utilities
- `chunk_pdf.py` – PDF chunking logic
- `create_vectorstore.py` – vector database creation helper
- `rag_test.py` – RAG workflow testing
- `search_test.py` – retrieval/search testing
- `embedding_test.py` – embedding checks
- `documents/` – uploaded PDF files
- `vectorstore/` – generated FAISS index files
- `requirements.txt` – project dependencies

## Requirements

Python 3.10+ is recommended.

## Setup

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Google API key:

```env
GOOGLE_API_KEY=your_api_key_here
```

> If your environment uses a different variable name in your setup, make sure the library receives the correct Google API credentials.

## Running the App
.\venv\Scripts\Activate.ps1
From the project root, run:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal in your browser.

## Workflow

1. Upload a PDF file in the app
2. Click the Process Document button
3. The app extracts text, splits it into chunks, and creates a FAISS vector store
4. Enter a legal question in the chat interface
5. The app retrieves relevant document chunks and generates an answer based on the document content

## Notes

- The app saves processed PDFs under the `documents/` folder
- The FAISS index is stored in the `vectorstore/` folder
- This project is intended for legal document analysis and research, not legal representation or advice

## Example Usage

Example questions:

- What was the key decision in this document?
- Which section covers the obligations mentioned?
- What are the timeline or deadlines stated in the document?

## License

This project does not include a specific license file. If you plan to share or distribute it publicly, add a license that matches your intended use.
