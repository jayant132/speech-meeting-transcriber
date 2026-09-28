# Speech Meeting Transcriber

A voice-based Minutes of Meeting (MoM) pipeline: upload a recorded meeting and
get back a speaker-labeled, timestamped transcript, speaking-time statistics,
and an AI-generated summary with key points, decisions, and action items.

Supports **English, Hindi, and Odia**, including meetings where speakers
switch between languages mid-conversation. Every model in the pipeline runs
locally — no paid or external LLM APIs are used at any stage.

## How It Works

```
Audio/Video Upload
      |
      v
ffmpeg  ->  normalize to 16kHz mono WAV
      |
      v
Silero VAD  ->  strip silence, isolate speech regions
      |
      v
pyannote.audio 3.1  ->  speaker diarization (Person 1, Person 2, ...)
      |
      v
Merge adjacent same-speaker turns  ->  see "Why Segments Are Merged" below
      |
      v
Per-segment language ID (facebook/mms-lid-126)
      |
      +--> en / hi  -->  faster-whisper (medium, int8), language forced explicitly
      |
      +--> or        -->  ai4bharat/indicwav2vec-odia + punctuation cleanup
      |
      +--> uncertain/unsupported  -->  faster-whisper auto-detect (fallback)
      |
      v
Merge transcript + speakers + timestamps + confidence  ->  speaking-time stats
      |
      v
Ollama (phi3.5)  ->  summary, key points, decisions, action items
      |
      v
SQLite (job + stage tracking)  <-->  FastAPI (/upload  /status/{id}  /result/{id})
```

Full architecture reasoning and technology justification:
[`docs/architecture.md`](docs/architecture.md).

## Why These Tools, Specifically

| Stage | Tool | Why |
|---|---|---|
| Preprocessing | ffmpeg | Reliable normalization of any input format to a consistent 16kHz mono WAV. |
| Speech detection | Silero VAD | Lightweight, fast, strips silence before the heavier stages run. |
| Diarization | pyannote.audio 3.1 | Purpose-built speaker segmentation model — not an LLM guessing who spoke when. Fully local. |
| Language ID | facebook/mms-lid-126 | Whisper's own language detection has no concept of Odia (it was never in Whisper's training data), so it can never route Odia audio correctly. MMS-LID is a dedicated audio classifier covering 126 languages, including Odia, run as its own step ahead of transcription. |
| ASR (English/Hindi) | faster-whisper, medium, int8 | CTranslate2-optimized Whisper reimplementation — same accuracy, much lighter on CPU/VRAM. Language is passed explicitly from MMS-LID's result rather than left to Whisper's own guess. |
| ASR (Odia) | ai4bharat/indicwav2vec-odia | Whisper cannot transcribe Odia at all. This is a dedicated Wav2Vec2 CTC model trained specifically on Odia speech. |
| Summarization | Ollama running phi3.5, locally | Fully local LLM inference, structured JSON output, with a guard that rejects a summary if it's just echoing the transcript back. |
| Storage | SQLite | Single-machine, sequential job processing — no case for a server-based database here. |
| API | FastAPI + BackgroundTasks | Async job handling without the operational overhead of Celery/Redis. |

## Why Segments Are Merged Before Transcription

Diarization produces many short turns, some under a second. Feeding an ASR
model isolated, context-free fragments this short causes it to hallucinate —
it "fills in" plausible-sounding words when it doesn't have enough audio to
be confident. This was directly observed during development: the same audio
produced accurate, coherent text when transcribed as one continuous file, and
garbled, nonsensical text when transcribed as many tiny diarized fragments.

The fix: adjacent turns from the **same speaker**, separated by a gap under 1
second, are merged into a single continuous segment (capped at ~28 seconds,
just under Whisper's own ~30-second training window) before transcription.
Turns from different speakers are never merged. `condition_on_previous_text`
is disabled to prevent repetition across segments, and `beam_size=5` is used
for more accurate decoding.

## Confidence Scores

Every transcribed segment includes a `confidence` value (0-1):
- **faster-whisper**: derived from the model's average log-probability per segment
- **IndicWav2Vec (Odia)**: mean of the CTC decoder's top token probability across the segment

In real testing, this proved to be a genuinely useful signal, not just a
reported number: correctly language-routed Odia segments scored 0.95-0.97,
while segments where language identification misfired and fell back to the
wrong model scored ~0.74-0.75 — the confidence score correctly flags lower
-quality output without any special-casing. See
[`docs/limitations.md`](docs/limitations.md) for the full finding.

## Job Stage Tracking

Processing a recording can take a few minutes, especially on CPU-only
hardware. Rather than `/status/{job_id}` showing a flat `pending` for the
entire duration, it reports the current pipeline stage:

```json
{"job_id": "...", "status": "pending", "stage": "transcribing"}
```

Possible `stage` values: `queued`, `normalizing_audio`, `diarizing`,
`transcribing`, `analyzing`, `summarizing`, `done`, `failed`. This gives a
caller visibility into whether a job is progressing or stuck, without
changing any existing response fields.

## Requirements

- Python 3.12
- [ffmpeg](https://ffmpeg.org/download.html) — on Windows, use a
  **full-shared** build, not `full_build` static, or torchaudio/torchcodec
  can fail to load its DLLs even with ffmpeg on PATH. This project decodes
  audio via an `ffmpeg` subprocess + `soundfile` specifically to avoid that
  dependency, but ffmpeg itself must still be reachable on PATH.
- [Ollama](https://ollama.com), with `phi3.5` pulled
- A Hugging Face account with a **Read** access token, with the gated terms
  accepted for:
  - [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)

## Setup

```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate
# macOS / Linux
source venv/bin/activate

pip install uv

# GPU (adjust the CUDA index for your setup), or omit this line to install
# CPU-only torch via requirements.txt instead
uv pip install torch --index-url https://download.pytorch.org/whl/cu118

uv pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
HF_TOKEN=your_huggingface_token_here
```

Pull the summarization model and start Ollama's server:

```bash
ollama pull phi3.5
ollama serve
```

## Running

```bash
uvicorn src.main:app --reload
```

Interactive API docs (also usable as a lightweight UI — see
[Interface](#interface) below): `http://127.0.0.1:8000/docs`

### Upload a recording

```bash
curl -X POST "http://127.0.0.1:8000/upload" -F "file=@samples/meeting_recording.mp3"
```

Returns a `job_id`. Processing runs as a background task.

### Check status

```bash
curl "http://127.0.0.1:8000/status/<job_id>"
```

### Get the result

Once `status` is `completed`:

```bash
curl "http://127.0.0.1:8000/result/<job_id>"
```

Returns the full structured output: speaker-labeled transcript with
timestamps, per-segment language, and confidence score; speaking-time
statistics per participant; and the generated meeting summary with key
points, decisions, and action items.

## Interface

No separate frontend is provided. FastAPI's built-in interactive
documentation at `/docs` allows uploading a file, checking status, and
viewing results directly in the browser — no code or command line required.
This is a deliberate choice, not an omission: the assignment explicitly asks
candidates to prioritize pipeline reliability over building a complex UI.

## Configuration

All settings live in `src/config.py` and can be overridden via environment
variables:

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `medium` | faster-whisper model size for English/Hindi |
| `WHISPER_COMPUTE_TYPE` | `int8` | Quantization — keep at `int8` for CPU/low-VRAM |
| `WHISPER_DEVICE` | `cpu` | Set to `cuda` only if your GPU/driver support it — see Known Issues |
| `ODIA_MODEL` | `ai4bharat/indicwav2vec-odia` | Odia-specific ASR model |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | |
| `SUMMARIZATION_MODEL` | `phi3.5` | Any Ollama-pulled model name works |
| `OLLAMA_HOST` | `http://localhost:11434` | |

## Testing

```bash
pytest tests/ -v
```

25 tests passing, 1 skipped by default (a real-model integration test). To
include it:

```bash
pytest tests/ --run-slow
```

Coverage:
- `test_analyze.py` — transcript merging and speaker statistics, including edge cases
- `test_diarize.py` — speaker label stabilization, missing/corrupted file handling (this test suite found and led to fixing a real unhandled-exception bug)
- `test_transcribe.py` — Odia text cleanup, language-routing logic, confidence score presence, extraction failure handling
- `test_pipeline.py` — full orchestration with all heavy stages mocked, verifying error handling at every stage

Full real end-to-end test results, including a 3-speaker, 3-language meeting
recording: [`docs/test_results.md`](docs/test_results.md).

## Project Structure

```
src/
  main.py         FastAPI routes: /upload, /status/{id}, /result/{id}
  config.py       Centralized configuration
  audio.py        ffmpeg normalization + Silero VAD
  diarize.py      Speaker diarization + same-speaker segment merging
  transcribe.py   Language ID + routed transcription + confidence scoring
  analyze.py      Transcript merging + speaker statistics
  summarize.py    Local LLM summary and action-item extraction
  pipeline.py     Orchestrates every stage, tracks stage + handles errors
  storage.py      SQLite job persistence + stage tracking
  errors.py       Custom exception types used across the pipeline
tests/            pytest suite (25 passing, 1 slow/skipped)
docs/
  architecture.md   Full technology reasoning and design decisions
  limitations.md    Known issues, honestly documented
  test_results.md   Real end-to-end run results
samples/          Sample meeting recordings used for testing
```

## Known Issues

Full details in [`docs/limitations.md`](docs/limitations.md). Summary:

- **Windows + torchcodec**: bypassed by decoding audio via `ffmpeg` subprocess + `soundfile` instead of torchaudio's default backend.
- **CUDA hang on older drivers**: an outdated NVIDIA driver caused faster-whisper to hang indefinitely on GPU init; `WHISPER_DEVICE=cpu` is used as a result.
- **MMS-LID per-segment consistency**: on real multilingual audio, ~4 of 6 Odia segments were correctly identified and routed; shorter segments (under ~5s) occasionally fell back to Whisper's auto-detection instead. Confidence scores correctly reflect this gap (0.95+ for correctly-routed segments vs. ~0.75 for misrouted ones).
- **Very short diarized segments** can still transcribe less accurately than longer continuous speech, even after merging — an inherent limitation of Whisper-family and CTC-based ASR models.

## Deliverables Checklist

- [x] Complete source code
- [x] README with setup and execution instructions (this file)
- [x] Sample meeting recordings (`samples/`)
- [x] Structured transcript output (per-segment speaker, timestamps, text, language, confidence)
- [x] Speaker-wise conversation statistics
- [x] Generated Minutes of Meeting summary with key points, decisions, action items
- [x] Architecture and technology choices explanation (`docs/architecture.md`)
- [x] Test results and known limitations (`docs/test_results.md`, `docs/limitations.md`)

## Bonus Items Implemented

- Automatic language detection (per-segment, via MMS-LID)
- Speaker confidence scores
- Asynchronous processing (FastAPI BackgroundTasks)
- Automated testing (26 tests)

## Technology & API Constraints

No paid or external LLM API is used anywhere in this pipeline. All models —
pyannote, faster-whisper, MMS-LID, IndicWav2Vec, and phi3.5 — run entirely
locally.