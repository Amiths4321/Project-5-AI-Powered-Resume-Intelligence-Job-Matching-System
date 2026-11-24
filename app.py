
import streamlit as st
import pymupdf

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.match_explainer import explain_match


st.set_page_config(
    page_title="Resume Intelligence",
    page_icon="📄",
    layout="wide"
)

MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def extract_resume_text(uploaded_file):
    filename = uploaded_file.name
    content = uploaded_file.getvalue()

    if filename.lower().endswith(".txt"):
        return content.decode("utf-8", errors="ignore")

    if filename.lower().endswith(".pdf"):
        document = pymupdf.open(
            stream=content,
            filetype="pdf"
        )
        text = "\n".join(page.get_text() for page in document)
        document.close()
        return text

    return ""


def calculate_similarity(resume_text, job_text, model):
    embeddings = model.encode([resume_text, job_text])

    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return round(float(score) * 100, 2)


st.title("📄 AI Resume Intelligence & Job Matching")
st.write(
    "Upload resumes, compare them against a job description, "
    "and generate AI-powered explanations."
)

st.divider()

st.subheader("1. Enter Job Description")

job_text = st.text_area(
    "Paste the complete job description",
    height=220,
    placeholder="Paste job requirements, skills and responsibilities..."
)

st.subheader("2. Upload Resumes")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT resumes",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.button("Analyze Resumes", type="primary"):
    if not job_text.strip():
        st.error("Please enter a job description.")

    elif not uploaded_files:
        st.error("Please upload at least one resume.")

    else:
        with st.spinner("Loading semantic embedding model..."):
            model = load_embedding_model()

        results = []

        progress = st.progress(0)

        for index, uploaded_file in enumerate(uploaded_files):
            try:
                resume_text = extract_resume_text(uploaded_file)

                if not resume_text.strip():
                    st.warning(
                        f"No readable text found in {uploaded_file.name}"
                    )
                    continue

                score = calculate_similarity(
                    resume_text,
                    job_text,
                    model
                )

                results.append({
                    "filename": uploaded_file.name,
                    "score": score,
                    "text": resume_text
                })

            except Exception as error:
                st.error(
                    f"Could not process {uploaded_file.name}: {error}"
                )

            progress.progress((index + 1) / len(uploaded_files))

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        st.session_state["results"] = results
        st.session_state["job_text"] = job_text

if "results" in st.session_state:
    results = st.session_state["results"]

    st.divider()
    st.subheader("3. Resume Matching Results")

    if not results:
        st.warning("No resumes could be analyzed.")

    for index, result in enumerate(results):
        with st.expander(
            f"{index + 1}. {result['filename']} — "
            f"{result['score']}% similarity",
            expanded=(index == 0)
        ):
            st.metric(
                "Semantic Similarity",
                f"{result['score']}%"
            )

            if st.button(
                "Generate AI Explanation",
                key=f"explain_{index}"
            ):
                with st.spinner("Qwen is analyzing the resume..."):
                    try:
                        explanation = explain_match(
                            result["text"],
                            st.session_state["job_text"],
                            result["score"]
                        )
                        st.markdown(explanation)

                    except Exception as error:
                        st.error(f"AI explanation failed: {error}")

    st.caption(
        "Similarity scores are experimental text-similarity measures, "
        "not hiring probabilities or validated suitability assessments. "
        "Always review candidate information manually."
    )
