
import re

SKILL_LIST = [
    "Python",
    "SQL",
    "Pandas",
    "NumPy",
    "Excel",
    "Power BI",
    "Tableau",
    "Data Visualization",
    "Statistics",
    "Machine Learning",
    "Deep Learning",
    "TensorFlow",
    "PyTorch",
    "Scikit-learn",
    "NLP",
    "Natural Language Processing",
    "Computer Vision",
    "FastAPI",
    "Flask",
    "Streamlit",
    "AWS",
    "Azure",
    "Docker",
    "Git",
    "Linux",
    "Spark",
    "Hadoop",
    "R",
    "Java",
    "C++"
]


def extract_skills(text):
    """Extract known skills from resume text."""

    found_skills = []
    normalized_text = text.lower()

    for skill in SKILL_LIST:
        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"

        if re.search(pattern, normalized_text):
            found_skills.append(skill)

    return sorted(set(found_skills))


if __name__ == "__main__":
    from src.resume_parser import load_all_resumes

    resumes = load_all_resumes()

    for resume in resumes:
        skills = extract_skills(resume["text"])

        print("\nResume:", resume["source"])
        print("Extracted skills:", skills)
        print("Total skills:", len(skills))
