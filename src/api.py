
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pymupdf

from src.match_explainer import explain_match
from src.database import initialize_database, save_analysis, get_history

app = FastAPI(
    title="Resume Intelligence API",
    description="Resume and job-description semantic matching API",
    version="1.0.0"
)

MODEL_NAME = "all-MiniLM-L6-v2"
model = None


@app.on_event("startup")
def startup():
    initialize_database()


def get_model():
    global model

    if model is None:
        model = SentenceTransformer(MODEL_NAME)

    return model


def extract_text(filename, content):
    filename = filename.lower()

    if filename.endswith(".txt"):
        return content.decode("utf-8", errors="ignore")

    if filename.endswith(".pdf"):
        document = pymupdf.open(
            stream=content,
            filetype="pdf"
        )
        text = "\n".join(
            page.get_text() for page in document
        )
        document.close()
        return text

    raise ValueError("Only PDF and TXT files are supported.")


def calculate_similarity(resume_text, job_text):
    embedding_model = get_model()

    embeddings = embedding_model.encode(
        [resume_text, job_text]
    )

    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return round(float(score) * 100, 2)


@app.get("/")
def home():
    return {
        "message": "Resume Intelligence API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze_resumes(
    files: list[UploadFile] = File(...),
    job_description: str = Form(...),
    include_explanation: bool = Form(False)
):
    if not job_description.strip():
        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty."
        )

    results = []

    for uploaded_file in files:
        try:
            content = await uploaded_file.read()

            resume_text = extract_text(
                uploaded_file.filename,
                content
            )

            if not resume_text.strip():
                raise ValueError("No readable resume text found.")

            score = calculate_similarity(
                resume_text,
                job_description
            )

            explanation = None

            if include_explanation:
                explanation = explain_match(
                    resume_text,
                    job_description,
                    score
                )

            save_analysis(
                uploaded_file.filename,
                score,
                job_description,
                explanation
            )

            results.append({
                "filename": uploaded_file.filename,
                "similarity_score": score,
                "explanation": explanation
            })

        except Exception as error:
            results.append({
                "filename": uploaded_file.filename,
                "error": str(error)
            })

    results.sort(
        key=lambda item: item.get(
            "similarity_score", -1
        ),
        reverse=True
    )

    return {"results": results}


@app.get("/history")
def analysis_history():
    return {"history": get_history()}
