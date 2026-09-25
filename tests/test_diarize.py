
import pytest
from src.diarize import _stabilize_speaker_labels, diarize
from src.errors import DiarizationError


def test_stabilize_labels_assigns_by_first_appearance():
    turns = [
        {"start": 0.0, "end": 1.0, "raw_speaker": "SPEAKER_01"},
        {"start": 1.0, "end": 2.0, "raw_speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 3.0, "raw_speaker": "SPEAKER_01"},
    ]
    result = _stabilize_speaker_labels(turns)
    assert result[0]["speaker"] == "Person 1"
    assert result[1]["speaker"] == "Person 2"
    assert result[2]["speaker"] == "Person 1"


def test_stabilize_labels_removes_raw_speaker_key():
    turns = [{"start": 0.0, "end": 1.0, "raw_speaker": "SPEAKER_00"}]
    result = _stabilize_speaker_labels(turns)
    assert "raw_speaker" not in result[0]


def test_stabilize_labels_empty_input():
    assert _stabilize_speaker_labels([]) == []


def test_diarize_raises_on_missing_file():
    with pytest.raises(DiarizationError):
        diarize("samples/does_not_exist.mp3")


def test_diarize_raises_on_corrupted_file(tmp_path):
    bad_file = tmp_path / "corrupted.mp3"
    bad_file.write_bytes(b"not a real audio file")
    with pytest.raises(DiarizationError):
        diarize(str(bad_file))


@pytest.mark.slow
def test_diarize_real_audio_produces_ordered_turns(sample_audio_path):
    result = diarize(sample_audio_path)
    assert len(result) > 0
    starts = [t["start"] for t in result]
    assert starts == sorted(starts)
    assert all(t["speaker"].startswith("Person") for t in result)