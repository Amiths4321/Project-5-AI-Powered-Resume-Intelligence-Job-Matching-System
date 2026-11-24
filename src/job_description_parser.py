
from pathlib import Path
from src.skill_extractor import extract_skills

JOB_DESCRIPTION_PATH = Path("data/raw/job_description.txt")


def parse_job_description():
    if not JOB_DESCRIPTION_PATH.exists():
        raise FileNotFoundError(
            f"Job description not found: {JOB_DESCRIPTION_PATH}"
        )

    text = JOB_DESCRIPTION_PATH.read_text(encoding="utf-8")
    skills = extract_skills(text)

    return {
        "job_description": text,
        "required_skills": skills
    }


if __name__ == "__main__":
    result = parse_job_description()

    print("\nJob Description Skills:")
    for skill in result["required_skills"]:
        print(f"- {skill}")

    print(f"\nTotal required skills: {len(result['required_skills'])}")
