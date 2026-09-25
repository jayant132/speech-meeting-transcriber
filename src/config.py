import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

WHISPER_MODEL = "medium"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_DEVICE = "cuda"

ODIA_MODEL = "ai4bharat/indicwav2vec-odia"

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
HF_TOKEN = os.getenv("HF_TOKEN")

SUMMARIZATION_MODEL = "phi3.5"
OLLAMA_HOST = "http://localhost:11434"

SAMPLE_RATE = 16000

DB_PATH = BASE_DIR / "storage.db"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

SUPPORTED_LANGUAGES = {"en", "hi", "or"}
