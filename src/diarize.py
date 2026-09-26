import os
import subprocess
import tempfile

import soundfile as sf
import torch
from pyannote.audio import Pipeline
from src.config import DIARIZATION_MODEL, HF_TOKEN
from src.errors import DiarizationError

_diarization_pipeline = None

_MAX_MERGE_GAP = 1.0
_MAX_MERGED_DURATION = 28.0


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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        os.remove(out_path)
        raise DiarizationError(f"Failed to process audio file: {input_path}") from e
    return out_path


def _merge_adjacent_same_speaker(turns: list[dict]) -> list[dict]:
    if not turns:
        return turns

    merged = [dict(turns[0])]
    for turn in turns[1:]:
        last = merged[-1]
        gap = turn["start"] - last["end"]
        span = turn["end"] - last["start"]

        if turn["speaker"] == last["speaker"] and gap <= _MAX_MERGE_GAP and span <= _MAX_MERGED_DURATION:
            last["end"] = turn["end"]
        else:
            merged.append(dict(turn))

    return merged


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
    stabilized = _stabilize_speaker_labels(turns)
    return _merge_adjacent_same_speaker(stabilized)


def _stabilize_speaker_labels(turns: list[dict]) -> list[dict]:
    label_map = {}
    for turn in turns:
        raw = turn["raw_speaker"]
        if raw not in label_map:
            label_map[raw] = f"Person {len(label_map) + 1}"
        turn["speaker"] = label_map[raw]
        del turn["raw_speaker"]

    return turns
