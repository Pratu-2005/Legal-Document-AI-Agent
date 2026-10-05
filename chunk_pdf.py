from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF location
pdf_path = "documents/Jagriti_Devi_Judgment.pdf"

# Load PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("Number of pages:", len(documents))

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Split document into chunks
chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))

# Display first 3 chunks
for i, chunk in enumerate(chunks[:3]):
    print("\n==============================")
    print("CHUNK", i + 1)
    print("==============================")
    print(chunk.page_content)
    print("\nMetadata:", chunk.metadata)