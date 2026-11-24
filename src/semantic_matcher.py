
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.resume_parser import load_all_resumes
from src.job_description_parser import parse_job_description
from src.match_explainer import explain_match

MODEL_NAME = "all-MiniLM-L6-v2"


def combine_resume_pages(resumes):
    combined = {}

    for resume in resumes:
        source = resume["source"]
        combined.setdefault(source, [])
        combined[source].append(resume["text"])

    return {
        source: "\n".join(pages)
        for source, pages in combined.items()
    }


def calculate_semantic_match(resume_text, job_text, model):
    embeddings = model.encode([resume_text, job_text])

    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return round(float(score) * 100, 2)


if __name__ == "__main__":
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    resumes = combine_resume_pages(load_all_resumes())
    job = parse_job_description()
    job_text = job["job_description"]

    for source, resume_text in resumes.items():
        score = calculate_semantic_match(
            resume_text, job_text, model
        )

        print(f"\n{'=' * 50}")
        print(f"Resume: {source}")
        print(f"Semantic similarity: {score}%")
        print("\nQwen AI Match Explanation:")

        try:
            explanation = explain_match(
                resume_text, job_text, score
            )
            print(explanation)
        except Exception as error:
            print(f"AI explanation failed: {error}")
