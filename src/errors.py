class PipelineError(Exception):
    pass


class UnsupportedFileError(PipelineError):
    pass


class CorruptAudioError(PipelineError):
    pass


class UnsupportedLanguageError(PipelineError):
    pass


class TranscriptionError(PipelineError):
    pass


class DiarizationError(PipelineError):
    pass
