import gc
import torch
from src.audio import normalize_audio, get_speech_segments
from src.diarize import diarize
from src.transcribe import transcribe_segment, detect_meeting_language
from src.analyze import merge_transcript, compute_speaker_stats
from src.summarize import generate_summary
from src.storage import update_job_status
from src.config import UPLOAD_DIR
from src.errors import PipelineError


def _release_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def run_pipeline(job_id: str, input_path: str):
    try:
        wav_path = normalize_audio(input_path, str(UPLOAD_DIR))
        get_speech_segments(wav_path)

        diarized_turns = diarize(wav_path)
        _release_gpu()

        meeting_language = detect_meeting_language(wav_path)

        transcribed_segments = [
            transcribe_segment(wav_path, turn["start"], turn["end"], meeting_language)
            for turn in diarized_turns
        ]
        _release_gpu()

        merged_transcript = merge_transcript(diarized_turns, transcribed_segments)
        speaker_stats = compute_speaker_stats(merged_transcript)

        summary = generate_summary(merged_transcript)

        result = {
            "transcript": merged_transcript,
            "speaker_stats": speaker_stats,
            "summary": summary,
        }
        update_job_status(job_id, "completed", result=result)

    except PipelineError as e:
        update_job_status(job_id, "failed", error=str(e))
    except Exception as e:
        update_job_status(job_id, "failed", error=f"Unexpected error: {e}")
