import ollama
import faiss
import pickle
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load stored data from ingestion
INDEX_PATH = "vector_store/index.faiss"
CHUNKS_PATH = "vector_store/chunks.pkl"
META_PATH = "app/metadata.json"

# Load vector index & data
index = faiss.read_index(INDEX_PATH)
chunks = pickle.load(open(CHUNKS_PATH, "rb"))
metadata = json.load(open(META_PATH, "r"))

EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Retrieval 
def retrieve(query, k=3):
    query_vec = EMB_MODEL.encode([query]).astype("float32")
    distances, ids = index.search(query_vec, k)
    return ids[0]


# Format Citations
def format_citations(ids):
    citations = []
    for cid in ids:
        meta = metadata[cid]
        citations.append(f"{meta['file']} - Page {meta['page']}")
    return citations


# Asking the LLM
def ask(query, debug=False):
    top_ids = retrieve(query, k=2)
    retrieved_chunks = [chunks[i] for i in top_ids]

    context = "\n\n".join(
        f"Source: {metadata[i]['file']} Page {metadata[i]['page']}\n{chunks[i]}"
        for i in top_ids
    )

    prompt = f"""
You are an aviation expert AI assistant.

Answer ONLY using the context below.

STRICT RULES:
- If the answer is NOT directly supported by the context, reply EXACTLY:
"This information is not available in the provided document(s)."
- DO NOT mention anything else.
- DO NOT say you are an AI model.
- DO NOT apologize.
- DO NOT explain why you cannot answer.
- DO NOT reference missing information.
- DO NOT add extra text.

Your entire answer MUST be only one of these two:
1. A grounded answer based only on the provided context text.
2. Exactly: "This information is not available in the provided document(s)."

------------------ CONTEXT ------------------
{context}
---------------------------------------------

Question: {query}

Answer:
"""


    response = ollama.chat(
    model="llama2:7b",  
    messages=[
        {"role": "system", "content": "Respond concisely and ONLY use provided context."},
        {"role": "user", "content": prompt}
    ],
    options={
        "temperature": 0.0,
        "num_ctx": 1024,
        "top_k": 20,
        "reset": True
    })
    answer = response["message"]["content"]
    
    return {
        "answer": answer.strip(),
        "citations": format_citations(top_ids),
        "retrieved_chunks (debug)": retrieved_chunks if debug else None
    }
