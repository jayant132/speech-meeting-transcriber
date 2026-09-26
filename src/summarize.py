import json
import re
import requests
from src.config import SUMMARIZATION_MODEL, OLLAMA_HOST
from src.errors import PipelineError

_PROMPT_TEMPLATE = """You are a meeting-minutes assistant. Read the transcript below and produce a genuine summary in your own words. Do not copy transcript lines verbatim, do not repeat speaker labels as JSON keys, do not restate this instruction.

Transcript:
{transcript}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "action_items": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "key_points", "decisions", "action_items"],
}

_MIN_TRANSCRIPT_WORDS_FOR_ECHO_CHECK = 40
_WORD_OVERLAP_THRESHOLD = 0.75


def _format_transcript(merged_transcript: list[dict]) -> str:
    lines = [f"{entry['speaker']}: {entry['text']}" for entry in merged_transcript]
    return "\n".join(lines)


def _extract_json(raw_output: str) -> dict:
    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise PipelineError(f"Model did not return valid JSON: {raw_output[:300]}")


def _word_overlap_ratio(summary: str, transcript_text: str) -> float:
    summary_words = set(re.findall(r"\w+", summary.lower()))
    transcript_words = set(re.findall(r"\w+", transcript_text.lower()))
    if not summary_words:
        return 0.0
    return len(summary_words & transcript_words) / len(summary_words)


def _looks_like_echo(result: dict, transcript_text: str) -> bool:
    summary = result.get("summary", "")
    if not summary:
        return True

    transcript_word_count = len(re.findall(r"\w+", transcript_text))
    if transcript_word_count < _MIN_TRANSCRIPT_WORDS_FOR_ECHO_CHECK:
        return False

    if transcript_text[:40] and transcript_text[:40] in summary:
        return True

    return _word_overlap_ratio(summary, transcript_text) > _WORD_OVERLAP_THRESHOLD


def generate_summary(merged_transcript: list[dict]) -> dict:
    transcript_text = _format_transcript(merged_transcript)
    prompt = _PROMPT_TEMPLATE.format(transcript=transcript_text)

    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": SUMMARIZATION_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": _RESPONSE_SCHEMA,
            "options": {"temperature": 0.1},
        },
        timeout=180,
    )

    if response.status_code != 200:
        raise PipelineError(f"Summarization failed: {response.text}")

    raw_output = response.json()["response"].strip()
    result = _extract_json(raw_output)

    if _looks_like_echo(result, transcript_text):
        raise PipelineError("Summarizer echoed the transcript instead of summarizing it")

    return result
