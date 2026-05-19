import os
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# For single image analysis with optional user query
def generate_summary_from_land_cover(percentages: dict, image_path: str, user_question: str = None):
    readable = ", ".join([f"{k.replace('_', ' ')}: {v}%" for k, v in percentages.items()])
    
    if user_question:
        prompt = f"""
You are a satellite image analyst. Based on the following land cover percentages and the image, answer the user's question clearly.

Land cover data: {readable}

User question: {user_question}

Answer:
"""
    else:
        prompt = f"""
You are a satellite image analyst. Based on the following land cover percentages and the satellite image, write a clear summary.

Land cover distribution: {readable}

Your summary:
"""

    image = Image.open(image_path).resize((128, 128))
    response = model.generate_content([prompt, image])
    return response.text

# For comparing two images
def generate_change_summary(report1: dict, report2: dict):
    def to_str(report):
        return ", ".join([f"{k.replace('_', ' ')}: {v:.2f}%" for k, v in report.items()])

    readable1 = to_str(report1)
    readable2 = to_str(report2)

    prompt = f"""
You are a satellite imagery analyst. Below are two land cover reports for the same region captured at different times.

Report for Image 1 (Earlier): {readable1}
Report for Image 2 (Later): {readable2}

Compare and summarize the major changes in land use and cover.
"""

    response = model.generate_content(prompt)
    return response.text
