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