import subprocess
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from src.config import SAMPLE_RATE
from src.errors import UnsupportedFileError, CorruptAudioError

_vad_model = None
_vad_utils = None


def _load_vad():
    global _vad_model, _vad_utils
    if _vad_model is None:
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
    return _vad_model, _vad_utils


def normalize_audio(input_path: str, output_dir: str) -> str:
    input_path = Path(input_path)
    if not input_path.exists():
        raise UnsupportedFileError(f"File not found: {input_path}")

    output_path = Path(output_dir) / f"{input_path.stem}.wav"
    command = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0 or not output_path.exists():
        stderr_tail = result.stderr.decode(errors="ignore")[-500:] if result.stderr else "no ffmpeg output"
        raise CorruptAudioError(
            f"Failed to process '{input_path.name}' (ffmpeg exit code {result.returncode}): {stderr_tail}"
        )

    return str(output_path)


def get_speech_segments(wav_path: str) -> list[dict]:
    model, utils = _load_vad()
    get_speech_timestamps = utils[0]

    audio, sr = sf.read(wav_path, dtype="float32")
    wav = torch.from_numpy(audio)
    timestamps = get_speech_timestamps(wav, model, sampling_rate=sr)

    if not timestamps:
        raise CorruptAudioError("No speech detected in audio")

    return [
        {"start": ts["start"] / sr, "end": ts["end"] / sr}
        for ts in timestamps
    ]