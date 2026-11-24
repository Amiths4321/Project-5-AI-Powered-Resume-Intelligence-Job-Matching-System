
import os
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


def explain_match(resume_text, job_text, score):
    if not OLLAMA_URL or not OLLAMA_MODEL:
        raise ValueError("OLLAMA_URL or OLLAMA_MODEL is missing from .env")

    prompt = f"""
You are a resume-job matching assistant.

Compare the resume with the job description.

Semantic similarity score: {score}%

Resume:
{resume_text[:4000]}

Job Description:
{job_text[:3000]}

Provide:
1. Overall match summary
2. Relevant strengths
3. Missing or weakly evidenced skills
4. Suggestions for improving the resume

Do not invent experience or qualifications.
Treat the similarity score as a rough signal, not a hiring probability.
"""

    response = requests.post(
        f"{OLLAMA_URL.rstrip('/')}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()
    return response.json()["response"]


if __name__ == "__main__":
    print("Match explainer module ready.")
