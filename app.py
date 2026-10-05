import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Legal Document AI Agent",
    page_icon="⚖️",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("⚖️ Legal Document AI Agent")

st.write(
    "Upload a legal document and ask questions "
    "about its contents using AI."
)

st.info(
    "This system is designed for legal document analysis "
    "and educational/research purposes. It does not provide "
    "legal advice and should not replace a qualified legal professional."
)


# ==========================================
# PDF UPLOAD
# ==========================================

st.subheader("📄 Upload Legal Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# ==========================================
# PROCESS PDF
# ==========================================

if uploaded_file is not None:

    os.makedirs("documents", exist_ok=True)

    pdf_path = os.path.join(
        "documents",
        uploaded_file.name
    )

    # Save uploaded PDF
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(
        f"PDF uploaded successfully: {uploaded_file.name}"
    )

    # Process button
    if st.button("🔄 Process Document"):

        with st.spinner(
            "Processing document and creating vector database..."
        ):

            try:

                # ----------------------------------
                # Load PDF
                # ----------------------------------

                loader = PyPDFLoader(pdf_path)

                documents = loader.load()


                # ----------------------------------
                # Split PDF into chunks
                # ----------------------------------

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = text_splitter.split_documents(
                    documents
                )


                # ----------------------------------
                # Create Gemini embeddings
                # ----------------------------------

                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001"
                )


                # ----------------------------------
                # Create FAISS vector database
                # ----------------------------------

                vectorstore = FAISS.from_documents(
                    chunks,
                    embeddings
                )


                # ----------------------------------
                # Save vector database
                # ----------------------------------

                vectorstore.save_local(
                    "vectorstore"
                )


                # ----------------------------------
                # Display processing information
                # ----------------------------------

                st.success(
                    "✅ Document processed successfully!"
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Number of Pages",
                        len(documents)
                    )

                with col2:
                    st.metric(
                        "Number of Text Chunks",
                        len(chunks)
                    )

            except Exception as e:

                st.error(
                    f"Error while processing document: {e}"
                )


# ==========================================
# QUESTION ANSWERING
# ==========================================

st.divider()

st.subheader("💬 Ask a Question")

question = st.text_input(
    "Enter your question about the legal document:",
    placeholder="Example: What was the final decision?"
)


# ==========================================
# ASK QUESTION BUTTON
# ==========================================

if st.button("🤖 Ask Question"):

    # ----------------------------------
    # Check question
    # ----------------------------------

    if not question.strip():

        st.warning(
            "⚠️ Please enter a question."
        )


    # ----------------------------------
    # Check vector database
    # ----------------------------------

    elif not os.path.exists("vectorstore"):

        st.error(
            "❌ Please upload and process a document first."
        )


    else:

        with st.spinner(
            "🔍 Searching the legal document and generating answer..."
        ):

            try:

                # ==================================
                # CREATE EMBEDDINGS
                # ==================================

                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001"
                )


                # ==================================
                # LOAD FAISS DATABASE
                # ==================================

                vectorstore = FAISS.load_local(
                    "vectorstore",
                    embeddings,
                    allow_dangerous_deserialization=True
                )


                # ==================================
                # CREATE RETRIEVER
                # ==================================

                retriever = vectorstore.as_retriever(
                    search_kwargs={
                        "k": 4
                    }
                )


                # ==================================
                # GEMINI MODEL
                # ==================================

                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0
                )


                # ==================================
                # SHORT QUESTION HANDLING
                # ==================================

                if len(question.split()) <= 3:

                    query_prompt = f"""
You are a query understanding component
of a Legal Document AI system.

The user has entered a very short question.

Convert the short question into a clear,
complete search query that describes what
information should be retrieved from the
uploaded legal document.

Do NOT answer the question.

Only generate the improved search query.

Short question:
{question}

Return ONLY the improved search query.
"""

                    query_response = llm.invoke(
                        query_prompt
                    )

                    search_query = (
                        query_response.content.strip()
                    )

                else:

                    search_query = question


                # ==================================
                # RETRIEVE DOCUMENT CHUNKS
                # ==================================

                documents = retriever.invoke(
                    search_query
                )


                # ==================================
                # CREATE CONTEXT
                # ==================================

                context_parts = []

                for document in documents:

                    page = document.metadata.get(
                        "page"
                    )

                    if page is not None:

                        page_number = page + 1

                        context_parts.append(
                            f"[Page {page_number}]\n"
                            f"{document.page_content}"
                        )

                    else:

                        context_parts.append(
                            document.page_content
                        )


                context = "\n\n".join(
                    context_parts
                )


                # ==================================
                # RAG PROMPT
                # ==================================

                prompt = f"""
You are a Legal Document AI Assistant.

Your task is to answer the user's question
using ONLY the information available in the
provided legal document context.

IMPORTANT RULES:

1. Do not invent information.

2. Do not use outside knowledge.

3. Do not make assumptions that are not
   supported by the document.

4. If the answer is not available in the
   provided context, clearly say:
   "The information is not available in
   the provided document."

5. Give a clear and concise answer.

6. Mention relevant page numbers when
   possible.

7. Do not provide legal advice.

8. The answer must be based on the uploaded
   document.

--------------------------------------------

LEGAL DOCUMENT CONTEXT:

{context}

--------------------------------------------

USER QUESTION:

{question}

--------------------------------------------

ANSWER:
"""


                # ==================================
                # GENERATE ANSWER
                # ==================================

                response = llm.invoke(
                    prompt
                )


                # ==================================
                # DISPLAY ANSWER
                # ==================================

                st.subheader(
                    "🤖 AI Answer"
                )

                st.write(
                    response.content
                )


                # ==================================
                # DISPLAY SEARCH QUERY
                # ==================================

                if len(question.split()) <= 3:

                    st.subheader(
                        "🔍 Search Query Used"
                    )

                    st.write(
                        search_query
                    )


                # ==================================
                # DISPLAY SOURCES
                # ==================================

                st.subheader(
                    "📄 Sources"
                )

                seen_pages = set()

                for document in documents:

                    page = document.metadata.get(
                        "page"
                    )

                    if page is not None:

                        page_number = page + 1

                        if page_number not in seen_pages:

                            st.write(
                                f"📄 Page {page_number}"
                            )

                            seen_pages.add(
                                page_number
                            )


            except Exception as e:

                st.error(
                    f"❌ Error while answering question: {e}"
                )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "⚖️ Legal Document AI Agent | "
    "RAG + Gemini + FAISS + LangChain"
)

