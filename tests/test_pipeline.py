from unittest.mock import patch
from src.pipeline import run_pipeline
from src.errors import DiarizationError


@patch("src.pipeline.update_job_status")
@patch("src.pipeline.generate_summary")
@patch("src.pipeline.compute_speaker_stats")
@patch("src.pipeline.merge_transcript")
@patch("src.pipeline.transcribe_segment")
@patch("src.pipeline.diarize")
@patch("src.pipeline.get_speech_segments")
@patch("src.pipeline.normalize_audio")
def test_run_pipeline_success_calls_all_stages_in_order(
    mock_normalize, mock_vad, mock_diarize, mock_transcribe,
    mock_merge, mock_stats, mock_summary, mock_update_status,
):
    mock_normalize.return_value = "fake.wav"
    mock_diarize.return_value = [{"start": 0, "end": 1, "speaker": "Person 1"}]
    mock_transcribe.return_value = {"text": "hello", "language": "en"}
    mock_merge.return_value = [{"speaker": "Person 1", "text": "hello"}]
    mock_stats.return_value = {"Person 1": {"total_duration": 1}}
    mock_summary.return_value = {"summary": "test summary"}

    run_pipeline("job1", "input.mp3")

    mock_normalize.assert_called_once()
    mock_diarize.assert_called_once_with("fake.wav")
    mock_summary.assert_called_once()
    mock_update_status.assert_called_once_with("job1", "completed", result={
        "transcript": mock_merge.return_value,
        "speaker_stats": mock_stats.return_value,
        "summary": mock_summary.return_value,
    })


@patch("src.pipeline.update_job_status")
@patch("src.pipeline.diarize")
@patch("src.pipeline.get_speech_segments")
@patch("src.pipeline.normalize_audio")
def test_run_pipeline_diarization_failure_marks_job_failed(
    mock_normalize, mock_vad, mock_diarize, mock_update_status,
):
    mock_normalize.return_value = "fake.wav"
    mock_diarize.side_effect = DiarizationError("no speakers")

    run_pipeline("job2", "input.mp3")

    mock_update_status.assert_called_once_with("job2", "failed", error="no speakers")


@patch("src.pipeline.update_job_status")
@patch("src.pipeline.normalize_audio")
def test_run_pipeline_unexpected_exception_marks_job_failed(
    mock_normalize, mock_update_status,
):
    mock_normalize.side_effect = RuntimeError("disk full")

    run_pipeline("job3", "input.mp3")

    args, kwargs = mock_update_status.call_args
    assert args[0] == "job3"
    assert args[1] == "failed"
    assert "disk full" in kwargs["error"]
