from builtins import Exception
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

# ----------------------------------------
# Configure Gemini API
# ----------------------------------------

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable not found."
    )

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

# ----------------------------------------
# Generate AI Medical Report
# ----------------------------------------

def generate_medical_report(prediction, confidence):

    prompt = f"""
You are an AI Medical Assistant.

A deep learning model analyzed a chest X-ray.

Prediction: {prediction}
Confidence: {confidence:.2f}%

Write a professional medical report in Markdown.

Rules:
- DO NOT wrap the response inside triple backticks (```).
- DO NOT return a markdown code block.
- Return plain Markdown only.

Use the following format:

# 🩺 AI Medical Report

## Prediction
- Disease: {prediction}
- Confidence: {confidence:.2f}%

## Possible Findings

## Possible Symptoms

## General Precautions

## Recommendation

## Disclaimer
State clearly that this is NOT a medical diagnosis and that the patient should consult a qualified doctor.

Keep the report under 180 words.
"""

    try:
        response = model.generate_content(prompt)

        report = response.text

        # Remove markdown code fences if Gemini returns them
        report = report.replace("```markdown", "")
        report = report.replace("```", "").strip()

        return report

    except Exception as e:
        return f"Error generating report: {e}"


# ----------------------------------------
# Test
# ----------------------------------------

if __name__ == "__main__":

    report = generate_medical_report(
        prediction="PNEUMONIA",
        confidence=96.72
    )

    print(report)