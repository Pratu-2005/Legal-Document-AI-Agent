from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Load Gemini embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Load existing FAISS database
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# Question
question = "What was the final decision of the court?"

print("\nQuestion:")
print(question)

# Search relevant chunks
results = retriever.invoke(question)

print("\n========== RELEVANT DOCUMENT CHUNKS ==========\n")

for i, document in enumerate(results):

    print(f"\n--- Result {i + 1} ---")

    print(document.page_content)

    page = document.metadata.get("page")

    if page is not None:
        print("\nPage:", page + 1)