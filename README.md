# Speech Meeting Transcriber

A voice-based **Minutes of Meeting (MoM)** pipeline. Upload a recorded meeting and get back:

- a speaker-labeled, timestamped transcript
- speaking-time statistics per participant
- an AI-generated summary with key points, decisions, and action items

Supports **English, Hindi, and Odia**, including meetings where speakers switch languages mid-conversation. Every model in the pipeline runs **locally**: no paid or external LLM APIs are used at any stage.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Why These Tools](#why-these-tools)
3. [Why Segments Are Merged Before Transcription](#why-segments-are-merged-before-transcription)
4. [Confidence Scores](#confidence-scores)
5. [Job Stage Tracking](#job-stage-tracking)
6. [Requirements](#requirements)
7. [Setup](#setup)
8. [Running the Application](#running-the-application)
9. [Using the API Directly (Optional)](#using-the-api-directly-optional)
10. [Configuration](#configuration)
11. [Testing](#testing)
12. [Project Structure](#project-structure)
13. [Known Issues](#known-issues)
14. [Deliverables Checklist](#deliverables-checklist)
15. [Bonus Items Implemented](#bonus-items-implemented)
16. [Technology and API Constraints](#technology-and-api-constraints)

---

## How It Works

```text
Audio / Video Upload  (via Streamlit UI or REST API)
      |
      v
ffmpeg  ->  normalize to 16 kHz mono WAV
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
Per-segment language ID  (facebook/mms-lid-126)
      |
      +--> English / Hindi  -->  faster-whisper (medium, int8), language forced explicitly
      |
      +--> Odia             -->  ai4bharat/indicwav2vec-odia + punctuation cleanup
      |
      +--> Uncertain / unsupported  -->  faster-whisper auto-detect (fallback)
      |
      v
Merge transcript + speakers + timestamps + confidence  ->  speaking-time stats
      |
      v
Ollama (phi3.5)  ->  summary, key points, decisions, action items
      |
      v
SQLite (job + stage tracking)  <-->  FastAPI (/upload, /status/{id}, /result/{id})
      ^
      |
Streamlit UI  (calls the FastAPI endpoints on the user's behalf)
```

Full architecture reasoning and technology justification: [`docs/architecture.md`](docs/architecture.md).

---

## Why These Tools

| Stage | Tool | Why |
|---|---|---|
| Preprocessing | ffmpeg | Reliable normalization of any input format to a consistent 16 kHz mono WAV. |
| Speech detection | Silero VAD | Lightweight and fast; strips silence before the heavier stages run. |
| Diarization | pyannote.audio 3.1 | Purpose-built speaker segmentation model, not an LLM guessing who spoke when. Fully local. |
| Language ID | facebook/mms-lid-126 | Whisper's own language detection has no concept of Odia (it was never in Whisper's training data), so it can never route Odia audio correctly. MMS-LID is a dedicated audio classifier covering 126 languages, including Odia, and runs as its own step ahead of transcription. |
| ASR (English / Hindi) | faster-whisper, `medium`, int8 | CTranslate2-optimized Whisper reimplementation: same accuracy, much lighter on CPU/VRAM. The language is passed explicitly from MMS-LID's result rather than left to Whisper's own guess. |
| ASR (Odia) | ai4bharat/indicwav2vec-odia | Whisper cannot transcribe Odia at all. This is a dedicated Wav2Vec2 CTC model trained specifically on Odia speech. |
| Summarization | Ollama running phi3.5 | Fully local LLM inference with structured JSON output, plus a guard that rejects a summary if it merely echoes the transcript back. |
| Storage | SQLite | Single-machine, sequential job processing; there is no case for a server-based database here. |
| API | FastAPI + BackgroundTasks | Async job handling without the operational overhead of Celery/Redis. |
| UI | Streamlit | Lets non-technical users upload a recording and read the results in the browser, with no command-line calls. |

---

## Why Segments Are Merged Before Transcription

Diarization produces many short turns, some under a second. Feeding an ASR model isolated, context-free fragments this short causes it to hallucinate: it "fills in" plausible-sounding words when it does not have enough audio to be confident. This was observed directly during development. The same audio produced accurate, coherent text when transcribed as one continuous file, and garbled, nonsensical text when transcribed as many tiny diarized fragments.

**The fix:**

- Adjacent turns from the **same speaker**, separated by a gap of **under 1 second**, are merged into a single continuous segment.
- Merged segments are capped at **~28 seconds**, just under Whisper's own ~30-second training window.
- Turns from **different speakers are never merged**.
- `condition_on_previous_text` is disabled to prevent repetition across segments.
- `beam_size=5` is used for more accurate decoding.

---

## Confidence Scores

Every transcribed segment includes a confidence value between 0 and 1:

- **faster-whisper:** derived from the model's average log-probability per segment.
- **IndicWav2Vec (Odia):** the mean of the CTC decoder's top-token probability across the segment.

In real testing this proved to be a genuinely useful signal, not just a reported number. Correctly language-routed Odia segments scored **0.95-0.97**, while segments where language identification misfired and fell back to the wrong model scored around **0.74-0.75**. The confidence score flags lower-quality output without any special-casing. See [`docs/limitations.md`](docs/limitations.md) for the full finding.

---

## Job Stage Tracking

Processing a recording can take a few minutes, especially on CPU-only hardware. Rather than showing a flat `pending` for the entire duration, `/status/{job_id}` reports the current pipeline stage:

```json
{"job_id": "...", "status": "pending", "stage": "transcribing"}
```

Possible `stage` values: `queued`, `normalizing_audio`, `diarizing`, `transcribing`, `analyzing`, `summarizing`, `done`, `failed`.

This gives a caller (and the Streamlit UI, which displays it live) visibility into whether a job is progressing or stuck, without changing any existing response fields.

---

## Requirements

- **Python 3.12**
- **ffmpeg** available on `PATH`
  - On Windows, use a **full-shared** build, not the `full_build` static build. Otherwise torchaudio/torchcodec can fail to load its DLLs even with ffmpeg on `PATH`. This project decodes audio via an ffmpeg subprocess plus `soundfile` specifically to avoid that dependency, but ffmpeg itself must still be reachable.
- **Ollama**, with the `phi3.5` model pulled
- A **Hugging Face account** with a *Read* access token, and the gated model terms accepted for:
  - [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install uv

# GPU users: install CUDA-enabled torch first (adjust the CUDA index for your setup).
# CPU-only users: skip this line; requirements.txt installs CPU torch.
uv pip install torch --index-url https://download.pytorch.org/whl/cu118

uv pip install -r requirements.txt
```

`requirements.txt` includes Streamlit for the UI.

### 3. Configure your Hugging Face token

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
```

### 4. Pull the summarization model

```bash
ollama pull phi3.5
```

---

## Running the Application

The application has **three components**. Start them **in this order**, each in its **own terminal** (activate the virtual environment in the second and third terminals).

### Step 1: Start the Ollama server

```bash
ollama serve
```

> If you see an "address already in use" error, Ollama is already running in the background (common with the Windows/macOS desktop app). That is fine; continue to the next step.

### Step 2: Start the FastAPI backend (Uvicorn)

```bash
uvicorn src.main:app --reload
```

The API is now available at `http://127.0.0.1:8000`. Interactive API docs are at `http://127.0.0.1:8000/docs`.

### Step 3: Start the Streamlit UI

```bash
streamlit run ui/app.py
```

Streamlit opens the app automatically in your browser at `http://localhost:8501`.

### Using the UI

1. Upload a meeting recording (audio or video).
2. Watch the live pipeline stage (`diarizing`, `transcribing`, `summarizing`, and so on).
3. When processing finishes, the page shows:
   - the speaker-labeled, timestamped transcript with per-segment language and confidence
   - speaking-time statistics per participant
   - the meeting summary: key points, decisions, and action items

> **Note:** The backend (Step 2) must be running before you upload a file, and Ollama (Step 1) must be running before the summarization stage is reached.

---

## Using the API Directly (Optional)

The Streamlit UI is a thin client over the REST API, so everything can also be done from the command line.

**Upload a recording**

```bash
curl -X POST "http://127.0.0.1:8000/upload" -F "file=@samples/meeting_recording.mp3"
```

Returns a `job_id`. Processing runs as a background task.

**Check status**

```bash
curl "http://127.0.0.1:8000/status/<job_id>"
```

**Get the result** (once status is `completed`)

```bash
curl "http://127.0.0.1:8000/result/<job_id>"
```

Returns the full structured output: the speaker-labeled transcript (timestamps, per-segment language, confidence), speaking-time statistics per participant, and the generated summary with key points, decisions, and action items.

---

## Configuration

All settings live in `src/config.py` and can be overridden through environment variables:

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `medium` | faster-whisper model size for English/Hindi |
| `WHISPER_COMPUTE_TYPE` | `int8` | Quantization; keep at `int8` for CPU/low-VRAM |
| `WHISPER_DEVICE` | `cpu` | Set to `cuda` only if your GPU/driver supports it (see [Known Issues](#known-issues)) |
| `ODIA_MODEL` | `ai4bharat/indicwav2vec-odia` | Odia-specific ASR model |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | Speaker diarization model |
| `SUMMARIZATION_MODEL` | `phi3.5` | Any Ollama-pulled model name works |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |

---

## Testing

```bash
pytest tests/ -v
```

**26 tests in total: 25 pass and 1 is skipped by default** (a real-model integration test). To include the slow test:

```bash
pytest tests/ --run-slow
```

**Coverage**

| File | What it covers |
|---|---|
| `test_analyze.py` | Transcript merging and speaker statistics, including edge cases |
| `test_diarize.py` | Speaker label stabilization, missing/corrupted file handling (this suite found and led to a fix for a real unhandled-exception bug) |
| `test_transcribe.py` | Odia text cleanup, language-routing logic, confidence score presence, extraction failure handling |
| `test_pipeline.py` | Full orchestration with all heavy stages mocked, verifying error handling at every stage |

Full real end-to-end results, including a 3-speaker, 3-language meeting recording: [`docs/test_results.md`](docs/test_results.md).

---

## Project Structure

```text
src/
  main.py         FastAPI routes: /upload, /status/{id}, /result/{id}
  config.py       Centralized configuration
  audio.py        ffmpeg normalization + Silero VAD
  diarize.py      Speaker diarization + same-speaker segment merging
  transcribe.py   Language ID + routed transcription + confidence scoring
  analyze.py      Transcript merging + speaker statistics
  summarize.py    Local LLM summary and action-item extraction
  pipeline.py     Orchestrates every stage, tracks stage, handles errors
  storage.py      SQLite job persistence + stage tracking
  errors.py       Custom exception types used across the pipeline
ui/
  app.py          Streamlit front end (upload, live stage, results view)
tests/            pytest suite (25 passing, 1 slow/skipped)
docs/
  architecture.md   Full technology reasoning and design decisions
  limitations.md    Known issues, honestly documented
  test_results.md   Real end-to-end run results
samples/          Official sample meeting recording used for the demo and testing
```

---

## Known Issues

Full details in [`docs/limitations.md`](docs/limitations.md). Summary:

- **Windows + torchcodec:** worked around by decoding audio via an ffmpeg subprocess and `soundfile` instead of torchaudio's default backend.
- **CUDA hang on older drivers:** an outdated NVIDIA driver caused faster-whisper to hang indefinitely on GPU initialization, so `WHISPER_DEVICE=cpu` is the default.
- **MMS-LID per-segment consistency:** on real multilingual audio, about 4 of 6 Odia segments were correctly identified and routed. Shorter segments (under ~5 s) occasionally fell back to Whisper's auto-detection instead. Confidence scores reflect this gap (0.95+ for correctly routed segments vs. ~0.75 for misrouted ones).
- **Very short diarized segments** can still transcribe less accurately than longer continuous speech, even after merging. This is an inherent limitation of Whisper-family and CTC-based ASR models.

---

## Deliverables Checklist

- [x] Complete source code
- [x] README with setup and execution instructions (this file)
- [x] Official sample meeting recording (`samples/meeting_recording.mp3`)
- [x] Structured transcript output (per-segment speaker, timestamps, text, language, confidence)
- [x] Speaker-wise conversation statistics
- [x] Generated Minutes of Meeting summary with key points, decisions, and action items
- [x] Web UI (Streamlit) so no command-line calls are needed
- [x] Architecture and technology choices explanation (`docs/architecture.md`)
- [x] Test results and known limitations (`docs/test_results.md`, `docs/limitations.md`)

---

## Bonus Items Implemented

- Automatic language detection (per-segment, via MMS-LID)
- Per-segment transcription confidence scores
- Asynchronous processing (FastAPI BackgroundTasks) with live stage tracking
- Automated testing (26 tests)
- Streamlit web interface

---

## Technology and API Constraints

No paid or external LLM API is used anywhere in this pipeline. All models (pyannote, faster-whisper, MMS-LID, IndicWav2Vec, and phi3.5) run entirely locally.
