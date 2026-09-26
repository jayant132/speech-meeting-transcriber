import os
import subprocess
import tempfile

import soundfile as sf
import torch
from pyannote.audio import Pipeline
from sklearn.cluster import AgglomerativeClustering
from src.config import DIARIZATION_MODEL, HF_TOKEN
from src.errors import DiarizationError

_diarization_pipeline = None


def _load_pipeline():
    global _diarization_pipeline
    if _diarization_pipeline is None:
        _diarization_pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL, token=HF_TOKEN
        )
    return _diarization_pipeline


def _to_wav(input_path: str) -> str:
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", out_path],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        if os.path.exists(out_path):
            os.remove(out_path)
        stderr_tail = e.stderr.decode(errors="ignore")[-500:] if e.stderr else "no ffmpeg output"
        raise DiarizationError(f"Failed to convert '{input_path}': {stderr_tail}") from e
    except FileNotFoundError as e:
        raise DiarizationError("ffmpeg is not installed or not on PATH") from e
    return out_path


def diarize(wav_path: str) -> list[dict]:
    pipeline = _load_pipeline()
    converted_path = _to_wav(wav_path)
    audio, sample_rate = sf.read(converted_path, dtype="float32")
    os.remove(converted_path)

    waveform = torch.from_numpy(audio).unsqueeze(0)
    result = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    annotation = result.speaker_diarization

    turns = []
    for turn, _, speaker_label in annotation.itertracks(yield_label=True):
        turns.append({
            "start": turn.start,
            "end": turn.end,
            "raw_speaker": speaker_label,
        })

    if not turns:
        raise DiarizationError("No speakers detected")

    turns.sort(key=lambda t: t["start"])
    return _stabilize_speaker_labels(turns)


def _stabilize_speaker_labels(turns: list[dict]) -> list[dict]:
    label_map = {}
    for turn in turns:
        raw = turn["raw_speaker"]
        if raw not in label_map:
            label_map[raw] = f"Person {len(label_map) + 1}"
        turn["speaker"] = label_map[raw]
        del turn["raw_speaker"]

    return turns
