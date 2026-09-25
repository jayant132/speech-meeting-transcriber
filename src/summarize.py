import json
import re
import requests
from src.config import SUMMARIZATION_MODEL, OLLAMA_HOST
from src.errors import PipelineError

_PROMPT_TEMPLATE = """You are analyzing a meeting transcript. Based on the transcript below, extract the following and respond ONLY with valid JSON, no other text:

{{
  "summary": "a concise 2-4 sentence summary of the meeting",
  "key_points": ["list of important discussion points"],
  "decisions": ["list of decisions made, if any"],
  "action_items": ["list of action items with responsible person if mentioned"]
}}

Transcript:
{transcript}
"""


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


def generate_summary(merged_transcript: list[dict]) -> dict:
    transcript_text = _format_transcript(merged_transcript)
    prompt = _PROMPT_TEMPLATE.format(transcript=transcript_text)

    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": SUMMARIZATION_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
        timeout=180,
    )

    if response.status_code != 200:
        raise PipelineError(f"Summarization failed: {response.text}")

    raw_output = response.json()["response"].strip()
    return _extract_json(raw_output)