from fastapi import FastAPI, Query
from ingest import ingest
from rag import ask

app = FastAPI(
    title="Aviation RAG System",
    description="RAG chatbot using LLaMA2 + FAISS + Aviation PDFs",
    version="1.0"
)

@app.get("/health")
def health():
    return {"status": "running", "model": "llama2"}

@app.post("/ingest")
def run_ingest():
    ingest()
    return {"message": " Ingestion completed. Vector store updated."}

@app.post("/ask")
def ask_question(
    q: str = Query(..., description="Enter your question"),
    debug: bool = False
):
    response = ask(q, debug=debug)
    return response
