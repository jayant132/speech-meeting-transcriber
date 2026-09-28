import pytest
from src.analyze import merge_transcript, compute_speaker_stats


def test_merge_transcript_combines_diarization_and_text():
    diarized_turns = [
        {"start": 0.0, "end": 1.0, "speaker": "Person 1"},
        {"start": 1.0, "end": 2.5, "speaker": "Person 2"},
    ]
    transcribed_segments = [
        {"text": "hello there", "language": "en"},
        {"text": "namaste", "language": "hi"},
    ]
    result = merge_transcript(diarized_turns, transcribed_segments)

    assert len(result) == 2
    assert result[0]["speaker"] == "Person 1"
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 1.0
    assert result[0]["text"] == "hello there"
    assert result[0]["language"] == "en"
    assert result[1]["speaker"] == "Person 2"
    assert result[1]["start"] == 1.0
    assert result[1]["end"] == 2.5
    assert result[1]["text"] == "namaste"
    assert result[1]["language"] == "hi"


def test_merge_transcript_empty_input():
    assert merge_transcript([], []) == []


def test_merge_transcript_mismatched_lengths_truncates_to_shorter():
    diarized_turns = [
        {"start": 0.0, "end": 1.0, "speaker": "Person 1"},
        {"start": 1.0, "end": 2.0, "speaker": "Person 2"},
    ]
    transcribed_segments = [{"text": "only one", "language": "en"}]

    result = merge_transcript(diarized_turns, transcribed_segments)

    assert len(result) == 1
    assert result[0]["speaker"] == "Person 1"
    assert result[0]["text"] == "only one"


def test_merge_transcript_preserves_order():
    diarized_turns = [
        {"start": 0.0, "end": 1.0, "speaker": "Person 2"},
        {"start": 1.0, "end": 2.0, "speaker": "Person 1"},
        {"start": 2.0, "end": 3.0, "speaker": "Person 2"},
    ]
    transcribed_segments = [
        {"text": "first", "language": "en"},
        {"text": "second", "language": "en"},
        {"text": "third", "language": "en"},
    ]

    result = merge_transcript(diarized_turns, transcribed_segments)

    assert [r["text"] for r in result] == ["first", "second", "third"]
    assert [r["speaker"] for r in result] == ["Person 2", "Person 1", "Person 2"]


def test_compute_speaker_stats_single_speaker():
    merged = [
        {"speaker": "Person 1", "start": 0.0, "end": 2.0, "text": "a", "language": "en"},
        {"speaker": "Person 1", "start": 2.0, "end": 3.0, "text": "b", "language": "en"},
    ]
    stats = compute_speaker_stats(merged)

    assert stats["Person 1"]["total_duration"] == 3.0
    assert stats["Person 1"]["segment_count"] == 2
    assert stats["Person 1"]["percentage"] == 100.0


def test_compute_speaker_stats_multiple_speakers_percentages_sum_to_100():
    merged = [
        {"speaker": "Person 1", "start": 0.0, "end": 3.0, "text": "a", "language": "en"},
        {"speaker": "Person 2", "start": 3.0, "end": 4.0, "text": "b", "language": "en"},
    ]
    stats = compute_speaker_stats(merged)

    total_pct = stats["Person 1"]["percentage"] + stats["Person 2"]["percentage"]
    assert total_pct == pytest.approx(100.0, abs=0.1)
    assert stats["Person 1"]["percentage"] == 75.0
    assert stats["Person 2"]["percentage"] == 25.0
    assert stats["Person 1"]["segment_count"] == 1
    assert stats["Person 2"]["segment_count"] == 1


def test_compute_speaker_stats_three_speakers_segment_counts():
    merged = [
        {"speaker": "Person 1", "start": 0.0, "end": 1.0, "text": "a", "language": "en"},
        {"speaker": "Person 2", "start": 1.0, "end": 2.0, "text": "b", "language": "en"},
        {"speaker": "Person 1", "start": 2.0, "end": 3.0, "text": "c", "language": "en"},
        {"speaker": "Person 3", "start": 3.0, "end": 4.0, "text": "d", "language": "en"},
    ]
    stats = compute_speaker_stats(merged)

    assert stats["Person 1"]["segment_count"] == 2
    assert stats["Person 2"]["segment_count"] == 1
    assert stats["Person 3"]["segment_count"] == 1
    assert stats["Person 1"]["total_duration"] == 2.0


def test_compute_speaker_stats_empty_transcript_returns_empty_dict():
    stats = compute_speaker_stats([])
    assert stats == {}


def test_compute_speaker_stats_zero_duration_turn_does_not_crash():
    merged = [{"speaker": "Person 1", "start": 5.0, "end": 5.0, "text": "", "language": "en"}]
    stats = compute_speaker_stats(merged)

    assert stats["Person 1"]["total_duration"] == 0.0
    assert stats["Person 1"]["segment_count"] == 1
    assert stats["Person 1"]["percentage"] == 0.0


def test_compute_speaker_stats_returns_plain_dict_not_defaultdict():
    merged = [{"speaker": "Person 1", "start": 0.0, "end": 1.0, "text": "a", "language": "en"}]
    stats = compute_speaker_stats(merged)

    assert type(stats) is dict
    assert "Person 2" not in stats