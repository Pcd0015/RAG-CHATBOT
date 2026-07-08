import os
import google.generativeai as genai
from dotenv import load_dotenv

# This loads your API key from the .env file
load_dotenv()

# Configure with your API Key
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

print("--- Models Available to your API Key ---")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(model.name)