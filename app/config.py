"""Configuration settings for ResearchPilot."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Provider & API Keys
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

# Model Names
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini" if MODEL_PROVIDER == "openai" else "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small" if MODEL_PROVIDER == "openai" else "models/text-embedding-004")

# Paths
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
PAPERS_DIR = DATA_DIR / "papers"
DOCUMENTS_DIR = DATA_DIR / "documents"
INDEX_DIR = DATA_DIR / "index"
