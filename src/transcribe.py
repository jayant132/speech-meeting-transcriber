from pathlib import Path
from faster_whisper import WhisperModel
from transformers import AutoModelForCTC, AutoProcessor
import torch
import soundfile as sf
from src.config import WHISPER_MODEL, WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, ODIA_MODEL, SAMPLE_RATE
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
        _odia_model = AutoModelForCTC.from_pretrained(ODIA_MODEL)
        _odia_processor = AutoProcessor.from_pretrained(ODIA_MODEL)
    return _odia_model, _odia_processor


def _extract_segment_audio(wav_path: str, start: float, end: float):
    audio, sr = sf.read(wav_path)
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    return audio[start_sample:end_sample], sr


def _transcribe_odia(audio, sample_rate: int) -> str:
    model, processor = _load_odia()
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
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


def transcribe_segment(wav_path: str, start: float, end: float) -> dict:
    try:
        audio, sr = _extract_segment_audio(wav_path, start, end)
    except Exception as e:
        raise TranscriptionError(f"Failed to extract segment: {e}")

    whisper = _load_whisper()
    segments, info = whisper.transcribe(audio, language=None)
    detected_language = info.language

    if detected_language == "or":
        text = _transcribe_odia(audio, sr)
    elif detected_language in ("en", "hi"):
        text = " ".join(s.text.strip() for s in segments)
    else:
        text = " ".join(s.text.strip() for s in segments)
        detected_language = "unsupported"

    return {"text": text.strip(), "language": detected_language}
