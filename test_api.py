from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key from .env
load_dotenv()

# Create Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Send a test question
response = llm.invoke(
    "Explain artificial intelligence in one simple sentence."
)

# Print the answer
print(response.content)