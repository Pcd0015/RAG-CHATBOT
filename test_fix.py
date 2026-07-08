import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# We use the new model name here
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash", 
    google_api_key=os.getenv("GEMINI_API_KEY")
)

print("Testing connection with gemini-3.5-flash...")
try:
    response = llm.invoke("Hi, are you working?")
    print("Success! Response:", response.content)
except Exception as e:
    print("Error occurred:", e)