from pathlib import Path
from faster_whisper import WhisperModel
from transformers import AutoModelForCTC, AutoProcessor
import torch
import torchaudio
import soundfile as sf
from src.config import WHISPER_MODEL, WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, ODIA_MODEL, SAMPLE_RATE, SUPPORTED_LANGUAGES, HF_TOKEN
from src.errors import UnsupportedLanguageError, TranscriptionError

_whisper_model = None
_odia_model = None
_odia_processor = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
        )
    return _whisper_model


def _load_odia():
    global _odia_model, _odia_processor
    if _odia_model is None:
        _odia_model = AutoModelForCTC.from_pretrained(ODIA_MODEL, token=HF_TOKEN)
        _odia_processor = AutoProcessor.from_pretrained(ODIA_MODEL, token=HF_TOKEN)
    return _odia_model, _odia_processor


def _extract_segment_audio(wav_path: str, start: float, end: float):
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    return audio[start_sample:end_sample], sr


def _transcribe_odia(audio, sample_rate: int) -> str:
    model, processor = _load_odia()
    if sample_rate != SAMPLE_RATE:
        audio = torchaudio.functional.resample(
            torch.tensor(audio, dtype=torch.float32), sample_rate, SAMPLE_RATE
        ).numpy()
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    prediction_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(prediction_ids)[0]
    return _cleanup_odia_text(text)


def _cleanup_odia_text(text: str) -> str:
    text = " ".join(text.split())
    if text and not text.endswith("।"):
        text += "।"
    return text


def detect_meeting_language(wav_path: str) -> str:
    whisper = _load_whisper()
    _, info = whisper.transcribe(wav_path, language=None)
    return info.language if info.language in SUPPORTED_LANGUAGES else "unsupported"


def transcribe_segment(wav_path: str, start: float, end: float, language: str) -> dict:
    try:
        audio, sr = _extract_segment_audio(wav_path, start, end)
    except Exception as e:
        raise TranscriptionError(f"Failed to extract segment: {e}")

    if language == "or":
        text = _transcribe_odia(audio, sr)
    elif language in ("en", "hi"):
        whisper = _load_whisper()
        segments, _ = whisper.transcribe(audio, language=language)
        text = " ".join(s.text.strip() for s in segments)
    else:
        whisper = _load_whisper()
        segments, _ = whisper.transcribe(audio, language=None)
        text = " ".join(s.text.strip() for s in segments)

    return {"text": text.strip(), "language": language}