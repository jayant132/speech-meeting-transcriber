import sqlite3
import json
import uuid
from datetime import datetime
from src.config import DB_PATH


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            filename TEXT,
            result TEXT,
            error TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def create_job(filename: str) -> str:
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO jobs (id, status, filename, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, "pending", filename, now, now),
    )
    conn.commit()
    conn.close()
    return job_id


def update_job_status(job_id: str, status: str, result: dict = None, error: str = None):
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE jobs SET status = ?, result = ?, error = ?, updated_at = ? WHERE id = ?",
        (status, json.dumps(result) if result else None, error, now, job_id),
    )
    conn.commit()
    conn.close()


def get_job(job_id: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()

    if row is None:
        return None

    job = dict(row)
    if job["result"]:
        job["result"] = json.loads(job["result"])
    return job
