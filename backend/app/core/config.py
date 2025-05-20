import os

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api/generate")
