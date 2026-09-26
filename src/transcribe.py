from pathlib import Path
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

    model, extractor = _load_lid()
    inputs = extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    raw_label = model.config.id2label[predicted_id]
    return _LID_LABEL_MAP.get(raw_label, "unsupported")


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

    language = _detect_language(audio, sr)

    if language == "or":
        text = _transcribe_odia(audio, sr)
    elif language in ("en", "hi"):
        whisper = _load_whisper()
        segments, _ = whisper.transcribe(audio, language=language)
        text = " ".join(s.text.strip() for s in segments)
    else:
        whisper = _load_whisper()
        segments, info = whisper.transcribe(audio, language=None)
        text = " ".join(s.text.strip() for s in segments)
        language = info.language if language == "uncertain" else "unsupported"

    return {"text": text.strip(), "language": language}
