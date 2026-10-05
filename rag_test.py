from dotenv import load_dotenv
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS

load_dotenv()

# --------------------------------
# 1. Load embeddings
# --------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# --------------------------------
# 2. Load FAISS database
# --------------------------------

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# --------------------------------
# 3. Create retriever
# --------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# --------------------------------
# 4. Load Gemini LLM
# --------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# --------------------------------
# 5. Ask question
# --------------------------------

question = "What was the final decision of the court?"

print("\nQuestion:")
print(question)

# Retrieve relevant documents
documents = retriever.invoke(question)

# --------------------------------
# 6. Create context
# --------------------------------

context = "\n\n".join(
    document.page_content
    for document in documents
)

# --------------------------------
# 7. Create RAG prompt
# --------------------------------

prompt = f"""
You are a Legal Document AI Assistant.

Answer the user's question using ONLY the
provided legal document context.

Rules:
1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not available in the context,
   say that the information is not available
   in the provided document.
4. Give a clear and concise answer.
5. Mention relevant page numbers when possible.

LEGAL DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

# --------------------------------
# 8. Generate answer
# --------------------------------

print("\nGenerating answer...")

response = llm.invoke(prompt)

print("\n========== AI ANSWER ==========\n")
print(response.content)

# --------------------------------
# 9. Show sources
# --------------------------------

print("\n========== SOURCES ==========\n")

seen_pages = set()

for document in documents:

    page = document.metadata.get("page")

    if page is not None:
        page_number = page + 1

        if page_number not in seen_pages:
            print("Page:", page_number)
            seen_pages.add(page_number)