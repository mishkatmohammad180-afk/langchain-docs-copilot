import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Make sure it's set in your .env file.")

# Model settings
LLM_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Vector store settings
CHROMA_PERSIST_DIR = "data/chroma_db"

# GitHub ingestion settings
GITHUB_REPO = "langchain-ai/docs"
DOCS_PATH = "src/oss/python"
