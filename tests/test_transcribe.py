from unittest.mock import patch, MagicMock
import numpy as np
from src.transcribe import _cleanup_odia_text, transcribe_segment


def test_cleanup_odia_text_adds_danda_if_missing():
    result = _cleanup_odia_text("ଏହା ଏକ ପରୀକ୍ଷା")
    assert result.endswith("।")


def test_cleanup_odia_text_keeps_existing_danda():
    result = _cleanup_odia_text("ଏହା ଏକ ପରୀକ୍ଷା।")
    assert result.count("।") == 1


def test_cleanup_odia_text_normalizes_whitespace():
    result = _cleanup_odia_text("ଏହା   ଏକ    ପରୀକ୍ଷା")
    assert "  " not in result


def test_cleanup_odia_text_empty_input_returns_empty():
    assert _cleanup_odia_text("") == ""


@patch("src.transcribe._extract_segment_audio")
@patch("src.transcribe._detect_language")
@patch("src.transcribe._load_whisper")
def test_transcribe_segment_routes_english_to_whisper(
    mock_load_whisper, mock_detect, mock_extract,
):
    mock_extract.return_value = (np.zeros(16000), 16000)
    mock_detect.return_value = "en"
    mock_segment = MagicMock()
    mock_segment.text = "hello world"
    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = ([mock_segment], MagicMock())
    mock_load_whisper.return_value = mock_whisper

    result = transcribe_segment("fake.wav", 0.0, 1.0)

    assert result["language"] == "en"
    assert "hello world" in result["text"]
    mock_whisper.transcribe.assert_called_once()


@patch("src.transcribe._extract_segment_audio")
@patch("src.transcribe._detect_language")
@patch("src.transcribe._transcribe_odia")
def test_transcribe_segment_routes_odia_to_indicwav2vec(
    mock_transcribe_odia, mock_detect, mock_extract,
):
    mock_extract.return_value = (np.zeros(16000), 16000)
    mock_detect.return_value = "or"
    mock_transcribe_odia.return_value = "ଏହା ଏକ ପରୀକ୍ଷା।"

    result = transcribe_segment("fake.wav", 0.0, 1.0)

    assert result["language"] == "or"
    mock_transcribe_odia.assert_called_once()


@patch("src.transcribe._extract_segment_audio")
def test_transcribe_segment_extraction_failure_raises(mock_extract):
    from src.errors import TranscriptionError
    import pytest

    mock_extract.side_effect = Exception("corrupt segment")

    with pytest.raises(TranscriptionError):
        transcribe_segment("fake.wav", 0.0, 1.0)
