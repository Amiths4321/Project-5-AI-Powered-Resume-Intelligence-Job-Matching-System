
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.resume_parser import load_all_resumes
from src.skill_extractor import extract_skills
from src.job_description_parser import parse_job_description


def calculate_match(resume_skills, job_skills):
    resume_text = " ".join(resume_skills)
    job_text = " ".join(job_skills)

    if not resume_text or not job_text:
        return 0.0

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_text])

    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(float(score) * 100, 2)


if __name__ == "__main__":
    resumes = load_all_resumes()
    job = parse_job_description()
    job_skills = job["required_skills"]

    print("\nResume Matching Results")
    print("=" * 40)

    for resume in resumes:
        skills = extract_skills(resume["text"])
        score = calculate_match(skills, job_skills)

        matched = sorted(set(skills) & set(job_skills))
        missing = sorted(set(job_skills) - set(skills))

        print(f"\nResume: {resume['source']}")
        print(f"Match score: {score}%")
        print(f"Matched skills: {matched}")
        print(f"Missing skills: {missing}")
