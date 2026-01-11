from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Resume metadata CSV (absolute path)
RESUME_METADATA_PATH = BASE_DIR / "notebook" / "resumes_metadata.csv"

# FAISS index path
FAISS_INDEX_PATH = BASE_DIR / "notebook" / "faiss_index" / "resume_index.faiss"