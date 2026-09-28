# Architecture

## Overview

This system processes a recorded meeting (audio or video) and produces a
structured, speaker-labeled transcript with timestamps, per-language
transcription, speaking-time statistics, and an AI-generated summary with key
points, decisions, and action items. It supports English, Hindi, and Odia,
including recordings where speakers switch between languages within the same
meeting. Every component runs locally — no paid or external LLM API is used
at any stage.

## Pipeline Flow

```
                          Audio / Video Upload
                                   |
                                   v
                    ffmpeg  ->  normalize to 16kHz mono WAV
                                   |
                                   v
                    Silero VAD  ->  strip silence, isolate speech
                                   |
                                   v
              pyannote.audio 3.1  ->  speaker diarization
                     (Person 1, Person 2, Person 3, ...)
                                   |
                                   v
              Merge adjacent same-speaker turns into
              longer continuous segments (see below)
                                   |
                                   v
              Per-segment language identification
                     (facebook/mms-lid-126)
                                   |
              +--------------------+--------------------+
              |                    |                     |
              v                    v                     v
       en / hi                   or               uncertain / unsupported
              |                    |                     |
              v                    v                     v
      faster-whisper       ai4bharat/               faster-whisper
      (medium, int8,       indicwav2vec-odia        auto-detect
      language forced)     + text cleanup           (fallback)
              |                    |                     |
              +--------------------+--------------------+
                                   |
                                   v
              Merge: speaker + timestamps + text +
                   language + confidence score
                                   |
                                   v
                 Compute speaking-time statistics
                        per participant
                                   |
                                   v
              Ollama (phi3.5, local)  ->  summary,
              key points, decisions, action items
                                   |
                                   v
              SQLite (job storage + stage tracking)
                        <---->
              FastAPI (/upload  /status/{id}  /result/{id})
```

## Technology Choices

| Stage | Tool | Reasoning |
|---|---|---|
| Preprocessing | ffmpeg | Industry-standard, reliable normalization of any input format (audio or video) to a consistent 16kHz mono WAV, which every downstream model expects. |
| Speech detection | Silero VAD | Lightweight, fast, single-model dependency. Removes silence before the far more expensive diarization and transcription stages process it, reducing both runtime and the chance of models hallucinating on dead air. |
| Speaker diarization | pyannote.audio 3.1 | A purpose-built speaker segmentation and embedding model, not a general-purpose language model guessing who spoke when. Runs fully locally from pretrained weights, requiring no API key — only a one-time Hugging Face terms acceptance to download the gated model. |
| Language identification | facebook/mms-lid-126 | Whisper's own language detection has no concept of Odia at all, since Odia was never part of its training data. Relying on Whisper's guess for routing would silently fail on Odia audio. MMS-LID is a dedicated audio-based language classifier covering 126 languages, including Odia, run as its own explicit step before any transcription happens. This was the single most important correctness fix in the whole pipeline. |
| Speech-to-text (English / Hindi) | faster-whisper, medium model, int8 quantization | A CTranslate2-optimized reimplementation of OpenAI's Whisper — same underlying accuracy, significantly faster and lighter on CPU and VRAM. The language is passed explicitly from MMS-LID's result rather than left to Whisper's own auto-detection, which measurably improves accuracy on short or code-switched segments. |
| Speech-to-text (Odia) | ai4bharat/indicwav2vec-odia | Whisper cannot transcribe Odia under any configuration. This is a Wav2Vec2 CTC model trained specifically on Odia speech data, used only for segments MMS-LID identifies as Odia. It fills a real capability gap rather than duplicating a language Whisper already covers. |
| Odia text cleanup | Custom post-processing | CTC-based models output raw, unpunctuated text, unlike Whisper's sequence-to-sequence output which already includes natural punctuation. A lightweight normalization pass (whitespace cleanup, terminal punctuation) keeps the final transcript visually consistent across both transcription paths. |
| Summarization and extraction | Ollama running phi3.5, locally | Fully local LLM inference with no external API dependency. Output is constrained to a JSON schema (summary, key points, decisions, action items), and a dedicated guard rejects any output that simply echoes the transcript back rather than genuinely summarizing it. |
| Job storage | SQLite | The system processes one meeting at a time, on one machine, with no concurrent-write requirement that would justify the operational overhead of a server-based database. A single file, atomic writes, and full SQL query support are sufficient for job status and result retrieval. |
| API layer | FastAPI with BackgroundTasks | Provides asynchronous job handling — upload returns immediately, processing continues in the background — without introducing the infrastructure weight of a message queue such as Celery or Redis, which this project's scale does not warrant. |

## Key Design Decisions

### Per-Segment Language Routing, Not Per-File

Language is identified independently for every diarized speaker turn, using a
dedicated audio classifier rather than a single guess for the whole
recording. This is what allows a meeting where one participant speaks
English, another responds in Hindi, and a third switches to Odia
mid-conversation to be handled correctly, rather than assuming a single
language applies to the entire file. This was validated directly on a real
three-speaker, three-language recording, documented in `test_results.md`.

### Dedicated Language Identification Instead of Relying on Whisper's Guess

Whisper's own language output is a side effect of how it decodes audio, not a
purpose-built classifier — it has no mechanism to output "Odia" because that
language was never part of its training data. Early testing confirmed this
directly: Whisper's auto-detection consistently misclassified Odia audio as
the nearest language it did recognize, silently producing incorrect output
rather than failing visibly. Introducing MMS-LID as an explicit
pre-transcription step, rather than trusting Whisper's internal guess, is
what made genuine multilingual routing possible.

### Same-Speaker Segment Merging Before Transcription

Diarization naturally produces many short speaker turns, some under a
second. Feeding a speech-to-text model isolated, context-free fragments this
short causes it to hallucinate: it fills in plausible-sounding words when it
lacks enough audio to be confident. This was directly observed during
development — identical audio produced accurate, coherent transcription when
processed as one continuous segment, and garbled, incorrect text when
processed as many tiny diarized fragments.

The resolution merges adjacent turns from the same speaker when the gap
between them is under one second, capping the merged segment at
approximately twenty-eight seconds — just under Whisper's own thirty-second
training window. Turns from different speakers are never merged, and long
pauses between same-speaker turns are preserved rather than bridged. This
single change was the most significant transcription-accuracy improvement
made to the pipeline.

### Confidence Scoring as a Diagnostic Signal, Not a Decorative Number

Every transcribed segment carries a confidence value between zero and one:
derived from Whisper's average log-probability for faster-whisper segments,
and from the mean top-token probability of the CTC decoder for Odia
segments. In real testing, this proved to be genuinely diagnostic rather
than cosmetic — segments where language identification correctly routed
Odia audio scored 0.95 or above, while segments where identification
misfired and fell back to the wrong model scored around 0.75. A consumer of
this API can use the confidence field to automatically flag low-quality
segments for review, without any additional logic beyond reading the number.

### Sequential Model Loading Under Constrained Hardware

Diarization, transcription, and summarization models are never held in GPU
memory simultaneously. GPU cache is explicitly released between pipeline
stages, allowing the full pipeline to run on hardware with limited VRAM by
processing one model's workload at a time rather than assuming all models
can coexist in memory.

### Job Stage Tracking for Long-Running Requests

A meeting recording can take several minutes to process, particularly on
CPU-only hardware. Rather than a caller seeing an undifferentiated "pending"
status for the entire duration, the job storage layer tracks which specific
pipeline stage is currently active — normalizing audio, diarizing,
transcribing, analyzing, or summarizing — and exposes it through the status
endpoint. This distinguishes a slow-but-progressing job from a genuinely
stuck one, using a single additional field with no changes to existing
response shapes.

### No External LLM Dependencies

Every model in this pipeline — pyannote, faster-whisper, MMS-LID,
IndicWav2Vec, and the local Ollama-served summarization model — runs
entirely on local compute. No component depends on a paid or external LLM
API, and no API key for any such service is required anywhere in the
system.

### Graceful Degradation Over Silent Failure

Missing input files, corrupted audio, and segments too short for language
identification to process reliably are each handled with specific,
identifiable exception types and defined fallback behavior, rather than
allowing an unhandled error to crash the pipeline or leave a job silently
stuck. Every stage of the pipeline reports failure through the same job
status mechanism a successful run uses, so a caller never has to
distinguish "still running" from "failed and stopped responding."