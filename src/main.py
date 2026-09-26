import logging
import shutil

from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException
from fastapi.concurrency import run_in_threadpool

from src.storage import init_db, create_job, get_job
from src.pipeline import run_pipeline
from src.config import UPLOAD_DIR
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Speech Meeting Transcriber")

init_db()
logger.info("Database initialized, application starting up")


@app.post("/upload")
async def upload_meeting(file: UploadFile, background_tasks: BackgroundTasks):
    saved_path = UPLOAD_DIR / file.filename
    try:
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except OSError:
        logger.error("Failed to save uploaded file '%s'", file.filename, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    job_id = create_job(file.filename)
    logger.info("Job %s created for file '%s'", job_id, file.filename)

    background_tasks.add_task(run_in_threadpool, run_pipeline, job_id, str(saved_path))
    logger.info("Job %s scheduled for processing", job_id)

    return {"job_id": job_id, "status": "pending"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        logger.warning("Status requested for unknown job %s", job_id)
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job["id"], "status": job["status"], "stage": job.get("stage")}


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    job = get_job(job_id)
    if job is None:
        logger.warning("Result requested for unknown job %s", job_id)
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        logger.info("Result requested for job %s but status is '%s'", job_id, job["status"])
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}")
    return job["result"]
