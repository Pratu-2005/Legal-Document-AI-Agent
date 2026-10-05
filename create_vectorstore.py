from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# --------------------------------
# 1. Load PDF
# --------------------------------

pdf_path = "documents/Jagriti_Devi_Judgment.pdf"

print("Loading PDF...")

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("Number of pages:", len(documents))


# --------------------------------
# 2. Split PDF into chunks
# --------------------------------

print("Splitting document into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# --------------------------------
# 3. Create Gemini embeddings
# --------------------------------

print("Creating embeddings...")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# --------------------------------
# 4. Create FAISS vector database
# --------------------------------

print("Creating FAISS vector database...")

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)


# --------------------------------
# 5. Save vector database
# --------------------------------

vectorstore.save_local("vectorstore")

print("FAISS vector database created successfully!")

print("Saved inside: vectorstore/")