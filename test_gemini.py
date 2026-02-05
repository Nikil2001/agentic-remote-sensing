import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

#List available models first
print("Available models:")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"  - {model.name}")

# Use gemini-2.0-flash (current available model)
model = genai.GenerativeModel("gemma-3-4b-it")
response = model.generate_content("Explain Devops in simple sentence")
print("\nResponse:", response.text)
