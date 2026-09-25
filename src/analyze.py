from collections import defaultdict


def merge_transcript(diarized_turns: list[dict], transcribed_segments: list[dict]) -> list[dict]:
    merged = []
    for turn, transcript in zip(diarized_turns, transcribed_segments):
        merged.append({
            "speaker": turn["speaker"],
            "start": turn["start"],
            "end": turn["end"],
            "text": transcript["text"],
            "language": transcript["language"],
        })
    return merged


def compute_speaker_stats(merged_transcript: list[dict]) -> dict:
    stats = defaultdict(lambda: {"total_duration": 0.0, "segment_count": 0})

    for entry in merged_transcript:
        duration = entry["end"] - entry["start"]
        stats[entry["speaker"]]["total_duration"] += duration
        stats[entry["speaker"]]["segment_count"] += 1

    total_duration = sum(s["total_duration"] for s in stats.values())

    for speaker_stats in stats.values():
        speaker_stats["percentage"] = (
            round((speaker_stats["total_duration"] / total_duration) * 100, 2)
            if total_duration > 0 else 0.0
        )

    return dict(stats)
