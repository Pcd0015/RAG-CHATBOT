"""
Central configuration for the RAG chatbot.
Updated for safe local and cloud deployment (July 2026).
"""
import os
import streamlit as st
from dotenv import load_dotenv

# Load local .env file if it exists
load_dotenv()

# --- API keys ---
# Robust loading: safely attempt to get from secrets, fallback to env
try:
    # This block prevents the FileNotFoundError when running locally without secrets.toml
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- Models ---
# gemini-3.5-flash is the current stable production default for 2026.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Paths ---
DATA_DIR = os.getenv("DATA_DIR", "data")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = "rag_chatbot_collection"

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", 4))

# --- Generation ---
MAX_TOKENS = 1024
TEMPERATURE = 0.2

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly using \
the provided context, which was retrieved from the user's own document collection.

Rules:
1. Answer ONLY using information found in the context below.
2. If the context does not contain enough information to answer, say so clearly \
instead of guessing or using outside knowledge.
3. Always cite which source file(s) you used, using the format.
4. Be concise and direct. Use bullet points for multi-part answers.
5. Do not fabricate quotes, numbers, or facts that are not present in the context.
"""