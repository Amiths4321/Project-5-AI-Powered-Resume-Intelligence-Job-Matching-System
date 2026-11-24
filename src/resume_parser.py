
from pathlib import Path
import pymupdf

RESUME_DIR = Path("documents")


def extract_resume_text(file_path):
    """Extract text from PDF or TXT resumes."""

    file_path = Path(file_path)

    if file_path.suffix.lower() == ".txt":
        text = file_path.read_text(encoding="utf-8").strip()

        return [{
            "text": text,
            "source": file_path.name,
            "page": 1
        }] if text else []

    extracted_pages = []

    with pymupdf.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                extracted_pages.append({
                    "text": text,
                    "source": file_path.name,
                    "page": page_number
                })

    return extracted_pages


def load_all_resumes():
    """Load PDF and TXT resumes."""

    all_pages = []

    for file_path in RESUME_DIR.iterdir():
        if file_path.suffix.lower() in [".pdf", ".txt"]:
            all_pages.extend(extract_resume_text(file_path))

    return all_pages


if __name__ == "__main__":
    resumes = load_all_resumes()

    print("Total extracted pages:", len(resumes))

    for resume in resumes:
        print("\nResume:", resume["source"])
        print("Text preview:", resume["text"][:300])
