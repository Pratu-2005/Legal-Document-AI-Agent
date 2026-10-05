from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

print("Loading Gemini embedding model...")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

print("Embedding model loaded successfully!")

text = "The accused was tried for an offence of murder."

vector = embeddings.embed_query(text)

print("Vector generated successfully!")
print("Vector length:", len(vector))
print("First 10 values:", vector[:10])