import json
import os
import time
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
POLL_INTERVAL_SECONDS = 3
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "samples" / "meeting_recording.mp3"

SOURCE_UPLOAD = "Upload a recording"
SOURCE_SAMPLE = "Use the sample meeting recording"

STAGE_LABELS = {
    "queued": "Queued",
    "normalizing_audio": "Normalizing audio",
    "diarizing": "Identifying speakers",
    "transcribing": "Transcribing speech",
    "analyzing": "Analyzing transcript",
    "summarizing": "Generating summary",
    "done": "Done",
    "failed": "Failed",
}

LANGUAGE_LABELS = {"en": "English", "hi": "Hindi", "or": "Odia"}


def format_time(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def submit_job(name: str, data: bytes) -> str:
    response = requests.post(
        f"{API_URL}/upload",
        files={"file": (name, data)},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["job_id"]


def wait_for_job(job_id: str, progress_slot) -> str:
    while True:
        response = requests.get(f"{API_URL}/status/{job_id}", timeout=30)
        response.raise_for_status()
        payload = response.json()
        stage = payload.get("stage") or "queued"
        progress_slot.info(f"Stage: {STAGE_LABELS.get(stage, stage)}")
        if payload["status"] in ("completed", "failed"):
            return payload["status"]
        time.sleep(POLL_INTERVAL_SECONDS)


def fetch_result(job_id: str) -> dict:
    response = requests.get(f"{API_URL}/result/{job_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def process(name: str, data: bytes):
    job_id = submit_job(name, data)
    progress_slot = st.empty()
    status = wait_for_job(job_id, progress_slot)
    if status == "failed":
        progress_slot.error("Processing failed. Check that the file is a valid audio or video recording.")
        return None
    progress_slot.success("Processing complete")
    return job_id, fetch_result(job_id)


def select_recording():
    if SAMPLE_PATH.is_file():
        source = st.radio("Recording source", [SOURCE_UPLOAD, SOURCE_SAMPLE], horizontal=True)
        if source == SOURCE_SAMPLE:
            return SAMPLE_PATH.name, SAMPLE_PATH.read_bytes()

    uploaded_file = st.file_uploader(
        "Meeting recording", type=["mp3", "wav", "m4a", "mp4", "mpeg", "ogg", "flac"]
    )
    if uploaded_file is None:
        return None
    return uploaded_file.name, uploaded_file.getvalue()


def render_summary(summary: dict):
    st.subheader("Summary")
    st.write(summary["summary"])

    columns = st.columns(3)
    sections = (
        ("Key Points", summary["key_points"]),
        ("Decisions", summary["decisions"]),
        ("Action Items", summary["action_items"]),
    )
    for column, (title, items) in zip(columns, sections):
        with column:
            st.markdown(f"**{title}**")
            if items:
                for item in items:
                    st.markdown(f"- {item}")
            else:
                st.caption("None identified")


def render_stats(speaker_stats: dict):
    st.subheader("Speaker Statistics")
    rows = [
        {
            "Speaker": speaker,
            "Speaking time (s)": round(stats["total_duration"], 1),
            "Segments": stats["segment_count"],
            "Share (%)": stats["percentage"],
        }
        for speaker, stats in speaker_stats.items()
    ]
    st.dataframe(rows, hide_index=True)


def render_transcript(transcript: list[dict]):
    st.subheader("Transcript")
    for entry in transcript:
        language = LANGUAGE_LABELS.get(entry["language"], entry["language"])
        confidence = entry.get("confidence")
        confidence_text = f" · confidence {confidence:.2f}" if confidence is not None else ""
        st.markdown(
            f"**{entry['speaker']}** `{format_time(entry['start'])} – {format_time(entry['end'])}` "
            f"· {language}{confidence_text}"
        )
        st.write(entry["text"])


def main():
    st.set_page_config(page_title="Speech Meeting Transcriber", layout="wide")
    st.title("Speech Meeting Transcriber")
    st.caption("Upload a meeting recording to get a speaker-labeled transcript, speaking statistics, and minutes.")

    recording = select_recording()

    if recording and st.button("Process meeting", type="primary"):
        st.session_state.pop("result", None)
        try:
            outcome = process(*recording)
        except requests.RequestException as error:
            st.error(f"Could not reach the API at {API_URL}: {error}")
            outcome = None
        if outcome:
            st.session_state["job_id"], st.session_state["result"] = outcome

    result = st.session_state.get("result")
    if result:
        render_summary(result["summary"])
        render_stats(result["speaker_stats"])
        render_transcript(result["transcript"])
        st.download_button(
            "Download result as JSON",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name=f"{st.session_state['job_id']}.json",
            mime="application/json",
        )


main()