import numpy as np
from pyannote.audio import Pipeline
from sklearn.cluster import AgglomerativeClustering
from src.config import DIARIZATION_MODEL, HF_TOKEN
from src.errors import DiarizationError

_diarization_pipeline = None


def _load_pipeline():
    global _diarization_pipeline
    if _diarization_pipeline is None:
        _diarization_pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL, use_auth_token=HF_TOKEN
        )
    return _diarization_pipeline


def diarize(wav_path: str) -> list[dict]:
    pipeline = _load_pipeline()
    diarization = pipeline(wav_path)

    turns = []
    embeddings = []
    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        turns.append({
            "start": turn.start,
            "end": turn.end,
            "raw_speaker": speaker_label,
        })

    if not turns:
        raise DiarizationError("No speakers detected")

    return _stabilize_speaker_labels(turns)


def _stabilize_speaker_labels(turns: list[dict]) -> list[dict]:
    raw_labels = sorted(set(t["raw_speaker"] for t in turns))
    label_map = {raw: f"Person {i + 1}" for i, raw in enumerate(raw_labels)}

    for turn in turns:
        turn["speaker"] = label_map[turn["raw_speaker"]]
        del turn["raw_speaker"]

    return turns
