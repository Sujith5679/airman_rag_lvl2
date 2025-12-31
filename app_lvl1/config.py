# app/config.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # fast + good for aviation docs
OLLAMA_MODEL = "llama3"               # change if you prefer qwen/phi
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
FAISS_PATH = "app/vectorstore.faiss"
