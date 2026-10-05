import os
from langchain_community.document_loaders import PyPDFLoader

pdf_path = "documents/Jagriti_Devi_Judgment.pdf"

# Check whether file exists
if not os.path.exists(pdf_path):
    print("PDF file not found!")
    print("Expected location:", os.path.abspath(pdf_path))
    exit()

print("PDF found!")
print("Reading PDF...")

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("Number of pages:", len(documents))

print("\n========== FIRST PAGE ==========\n")
print(documents[0].page_content)

print("\n========== METADATA ==========\n")
print(documents[0].metadata)