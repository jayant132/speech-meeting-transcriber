import shutil
from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException
from src.storage import init_db, create_job, get_job
from src.pipeline import run_pipeline
from src.config import UPLOAD_DIR

app = FastAPI(title="Speech Meeting Transcriber")

init_db()


@app.post("/upload")
async def upload_meeting(file: UploadFile, background_tasks: BackgroundTasks):
    saved_path = UPLOAD_DIR / file.filename
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = create_job(file.filename)
    background_tasks.add_task(run_pipeline, job_id, str(saved_path))

    return {"job_id": job_id, "status": "pending"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job["id"], "status": job["status"]}


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}")
    return job["result"]
