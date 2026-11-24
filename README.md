
# AI-Powered Resume Intelligence & Job Matching System

## Overview

An AI-assisted application that extracts information from resumes,
compares resumes with job descriptions, and generates explanations
using a locally hosted embedding model and an Ollama LLM.

## Features

- PDF and TXT resume extraction
- Rule-based skill extraction
- Job description parsing
- TF-IDF baseline matching
- Semantic similarity using sentence-transformers
- AI-generated match explanations using Ollama
- Streamlit dashboard
- FastAPI REST API
- SQLite analysis history

## Architecture

Resume PDF/TXT
      |
      v
Resume Parser
      |
      v
Skill Extraction
      |
      v
Text Embeddings
      |
      v
Semantic Similarity
      |
      v
Ollama Match Explanation
      |
      v
Streamlit Dashboard / FastAPI
      |
      v
SQLite History

## Technology Stack

- Python
- PyMuPDF
- Pandas
- Scikit-learn
- Sentence Transformers
- Streamlit
- FastAPI
- SQLite
- Ollama
- Qwen2.5-VL

## Setup

Clone the repository and enter the project directory.

Create a virtual environment:

    python -m venv .venv

Activate it on Windows:

    .venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements.txt

Create a local `.env` file using `.env.example` as a template.

Configure your Ollama endpoint and model in `.env`.

## Run Streamlit

    streamlit run app.py

## Run FastAPI

    python -m uvicorn src.api:app --reload --port 8001

Open:

    http://127.0.0.1:8001/docs

## Run Semantic Matching

    python -m src.semantic_matcher

## Run Baseline Matching

    python -m src.matcher

## Run Job Description Parser

    python -m src.job_description_parser

## Database

SQLite stores analysis history in:

    data/processed/resume_matching.db

## Evaluation Notes

The TF-IDF and semantic similarity scores are experimental
text-similarity measures.

They are not hiring probabilities, validated candidate rankings,
or measures of professional suitability.

AI-generated explanations may contain errors and must be reviewed.

## Privacy and Responsible Use

- Use synthetic or anonymized resumes for demonstrations.
- Do not commit real candidate documents or personal information.
- Keep environment variables and private endpoints out of GitHub.
- Human review is required for all employment-related decisions.
- This application is decision-support software, not an
  autonomous hiring system.

## Limitations

- Skill extraction uses a predefined keyword list.
- Scanned PDFs may require OCR.
- Similarity scores depend on resume length and wording.
- AI explanations may misinterpret or infer information.
- Results require human verification.

## Project Status

Learning and portfolio prototype.
