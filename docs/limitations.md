## Known Limitations

- **Audio decoding backend**: pyannote.audio 3.1 defaults to `torchcodec` for
  audio I/O. On Windows, torchcodec failed to load its FFmpeg-linked DLLs
  even with FFmpeg installed and on PATH (root cause: the installed FFmpeg
  build was `full_build` static, not `full_build-shared`, so the required
  .dll files for dynamic linking were absent). Rather than rebuild the FFmpeg
  toolchain, we bypassed torchcodec by pre-loading audio via
  `torchaudio.load()` and passing the raw waveform tensor directly to the
  pyannote pipeline. This avoids the torchcodec dependency entirely and is
  fully supported by pyannote's pipeline API.


  - **CUDA device hang**: `WHISPER_DEVICE="cuda"` caused faster-whisper
  (ctranslate2) to hang indefinitely during model initialization on this
  machine, rather than failing with a clear error — likely due to an
  outdated NVIDIA driver (462.30 / CUDA 11.2) incompatible with the
  installed ctranslate2/torch CUDA runtime expectations. Switched to
  `WHISPER_DEVICE="cpu"` to unblock development; GPU acceleration is a
  possible future improvement pending a driver update.