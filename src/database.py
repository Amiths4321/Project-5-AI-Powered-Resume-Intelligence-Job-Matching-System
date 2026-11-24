
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path("data/processed/resume_matching.db")


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS match_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_name TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                job_description TEXT NOT NULL,
                explanation TEXT,
                created_at TEXT NOT NULL
            )
        """)


def save_analysis(
    resume_name,
    score,
    job_description,
    explanation=None
):
    initialize_database()

    with get_connection() as connection:
        connection.execute("""
            INSERT INTO match_history (
                resume_name,
                similarity_score,
                job_description,
                explanation,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            resume_name,
            score,
            job_description,
            explanation,
            datetime.now(timezone.utc).isoformat()
        ))


def get_history(limit=50):
    initialize_database()

    with get_connection() as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute("""
            SELECT *
            FROM match_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [dict(row) for row in rows]
