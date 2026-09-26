# Known Limitations

## Audio Decoding Backend (Windows)
`torchaudio`/`torchcodec` (used internally by both Silero VAD and pyannote.audio)
failed to load its FFmpeg-linked DLLs on Windows, even with FFmpeg installed and
on PATH. Root cause: the installed FFmpeg build was `full_build` (static), not
`full_build-shared`, so the .dll files torchcodec needs for dynamic linking were
absent. Rather than rebuild the FFmpeg toolchain, audio decoding was routed
around torchcodec entirely: input files are converted via an `ffmpeg` subprocess
call to a normalized WAV, then read directly with `soundfile`. The raw waveform
array/tensor is passed straight to Silero VAD and pyannote's pipeline, avoiding
torchcodec at every stage.

## CUDA Device Hang
`WHISPER_DEVICE="cuda"` caused faster-whisper (ctranslate2) to hang indefinitely
during model initialization, rather than failing with a clear error. Root cause:
an outdated NVIDIA driver (462.30 / CUDA 11.2) incompatible with modern
ctranslate2/torch CUDA runtime expectations. `WHISPER_DEVICE="cpu"` is used for
all processing on this machine as a result. GPU acceleration remains possible on
this hardware pending a driver update, but was not pursued further given time
constraints.

## Whisper Does Not Support Odia
OpenAI's Whisper (and by extension faster-whisper) was trained on 99 languages,
none of which is Odia. Its own language-detection output can never return `or`,
so relying on it to route Odia audio silently fails — it guesses the nearest
language it knows instead of failing loudly. This was discovered during initial
testing and is the reason a dedicated language-identification step
(`facebook/mms-lid-126`) was added ahead of transcription, rather than relying on
Whisper's built-in detection for language routing.

## Language-ID on Very Short Segments
`facebook/mms-lid-126` requires a minimum audio length to run (its convolutional
layers need a minimum input size). Diarized turns shorter than ~0.25 seconds
cannot be reliably language-identified. These segments fall back to Whisper's
own language auto-detection rather than crashing the pipeline.

## Odia Transcription Verified, Not Independently Validated
Odia transcription (via `ai4bharat/indicwav2vec-odia`) is confirmed working
end-to-end: MMS-LID correctly identifies Odia segments, routes them away from
Whisper, and IndicWav2Vec produces valid Odia-script output. The output is a
**transcription**, not a translation — Odia speech is rendered in Odia script,
matching the assignment's requirement. Transcription accuracy has not been
independently verified against a native Odia speaker; this is a known
verification gap given project time constraints.

## Transcription Quality on Very Short Diarized Turns
Some diarized turns (particularly under ~1 second) produce noticeably rougher
transcription than longer turns, consistent with Whisper-family models
performing best on longer, contextual audio windows rather than isolated
fragments. This affects transcript readability on rapid back-and-forth exchanges
more than on longer speaking turns.

## Development Model Sizes vs. Final Configuration
`WHISPER_MODEL=tiny` and `SUMMARIZATION_MODEL=llama3.2:1b` were used during
early development for faster iteration on a CPU-only/low-VRAM machine. Final
testing and the results documented in `docs/test_results.md` use
`WHISPER_MODEL=medium` and `SUMMARIZATION_MODEL=phi3.5`, which is the intended
submission configuration.

## Summary Quality with Small Local Models
Ollama-based summarization (`phi3.5`, 3.8B parameters) occasionally over-states
opinions or hesitations as "decisions" when no explicit decision was made in the
transcript. An anti-echo guard prevents the model from simply restating the
transcript as a summary, but does not fully prevent this kind of
over-interpretation. A larger local model would likely improve this, at the cost
of slower inference on this hardware.

- **Short-segment language misdetection**: Whisper's language ID is less
  reliable on very short audio segments (under ~2 seconds), occasionally
  returning a language outside {en, hi, or} even when the transcribed text
  is clearly one of the supported languages. These segments are labeled
  "unsupported" but their transcribed text remains usable. A possible future
  improvement: fall back to the diarization turn's dominant/majority
  language when an individual short segment's LID confidence is low.