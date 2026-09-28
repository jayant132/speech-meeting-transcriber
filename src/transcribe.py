from pathlib import Path
import math
import torch
from faster_whisper import WhisperModel
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from transformers import AutoModelForCTC, AutoProcessor
import soundfile as sf
from src.config import WHISPER_MODEL, WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, ODIA_MODEL
from src.errors import TranscriptionError

_whisper_model = None
_odia_model = None
_odia_processor = None
_lid_model = None
_lid_extractor = None

_LID_LABEL_MAP = {"eng": "en", "hin": "hi", "ory": "or"}
_MIN_LID_SAMPLES = 4000
_SEGMENT_PADDING_SECONDS = 0.15
_BEAM_SIZE = 5


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


def _load_lid():
    global _lid_model, _lid_extractor
    if _lid_model is None:
        model_id = "facebook/mms-lid-126"
        _lid_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        _lid_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
    return _lid_model, _lid_extractor


def _detect_language(audio, sample_rate: int) -> str:
    if len(audio) < _MIN_LID_SAMPLES:
        return "uncertain"

    try:
        model, extractor = _load_lid()
        inputs = extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        raw_label = model.config.id2label[predicted_id]
        return _LID_LABEL_MAP.get(raw_label, "unsupported")
    except Exception as e:
        raise TranscriptionError(f"Language detection failed: {e}")


def _extract_segment_audio(wav_path: str, start: float, end: float):
    audio, sr = sf.read(wav_path)
    padded_start = max(0.0, start - _SEGMENT_PADDING_SECONDS)
    padded_end = min(len(audio) / sr, end + _SEGMENT_PADDING_SECONDS)
    start_sample = int(padded_start * sr)
    end_sample = int(padded_end * sr)
    return audio[start_sample:end_sample], sr


def _transcribe_odia(audio, sample_rate: int) -> tuple[str, float]:
    try:
        model, processor = _load_odia()
        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values.mean().item()
        prediction_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(prediction_ids)[0]
        return _cleanup_odia_text(text), round(confidence, 3)
    except Exception as e:
        raise TranscriptionError(f"Odia transcription failed: {e}")


def _cleanup_odia_text(text: str) -> str:
    text = " ".join(text.split())
    if text and not text.endswith("।"):
        text += "।"
    return text


def _transcribe_whisper(audio, language: str | None) -> tuple[str, str, float]:
    try:
        whisper = _load_whisper()
        segments, info = whisper.transcribe(
            audio,
            language=language,
            beam_size=_BEAM_SIZE,
            condition_on_previous_text=False,
        )
        segments = list(segments)
        text = " ".join(s.text.strip() for s in segments)
        detected_language = language or info.language

        if segments:
            avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)
            confidence = round(math.exp(avg_logprob), 3)
        else:
            confidence = 0.0

        return text, detected_language, confidence
    except Exception as e:
        raise TranscriptionError(f"Whisper transcription failed: {e}")


def transcribe_segment(wav_path: str, start: float, end: float) -> dict:
    try:
        audio, sr = _extract_segment_audio(wav_path, start, end)
    except Exception as e:
        raise TranscriptionError(f"Failed to extract segment: {e}")

    language = _detect_language(audio, sr)

    if language == "or":
        text, confidence = _transcribe_odia(audio, sr)
    elif language in ("en", "hi"):
        text, language, confidence = _transcribe_whisper(audio, language)
    else:
        text, detected, confidence = _transcribe_whisper(audio, None)
        language = detected if language == "uncertain" else "unsupported"

    return {"text": text.strip(), "language": language, "confidence": confidence}
