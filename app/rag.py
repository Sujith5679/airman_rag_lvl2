import ollama
import faiss
import pickle
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sentence_transformers.cross_encoder import CrossEncoder

# LOAD DATA & MODELS
INDEX_PATH = "vector_store/index.faiss"
CHUNKS_PATH = "vector_store/chunks.pkl"
META_PATH = "app/metadata.json"
BM25_PATH = "vector_store/bm25_index.pkl"

index = faiss.read_index(INDEX_PATH)
chunks = pickle.load(open(CHUNKS_PATH, "rb"))
metadata = json.load(open(META_PATH, "r"))

EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

with open(BM25_PATH, "rb") as f:
    bm25 = pickle.load(f)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# QUERY EXPANSION FOR BETTER SEARCH
EXPANSIONS = {
    "turbulence": ["wind shear", "mechanical turbulence", "thermal turbulence", "convective activity", "instability"],
    "dew point": ["humidity", "condensation", "saturation", "wet bulb"],
    "lift": ["aerodynamics", "airflow", "pressure differential"]
}

def expand_query(query: str):
    q = query.lower()
    for key, terms in EXPANSIONS.items():
        if key in q:
            query += " " + " ".join(terms)
    return query



#  HYBRID RETRIEVAL
def hybrid_retrieve(query, k=3):

    query = expand_query(query) 
    query_vec = EMB_MODEL.encode([query]).astype("float32")

    _, ids = index.search(query_vec, k)
    vector_chunks = [{"id": i, "text": chunks[i]} for i in ids[0]]

    tokenized = query.lower().split()
    scores = bm25.get_scores(tokenized)
    top_ids = np.argsort(scores)[::-1][:k]
    keyword_chunks = [{"id": i, "text": chunks[i]} for i in top_ids]

    candidates = {c["id"]: c for c in vector_chunks + keyword_chunks}
    candidates = list(candidates.values())

    ranking_pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(ranking_pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    filtered = [item[1] for item in ranked if len(item[1]["text"].split()) > 40]

    return filtered[:2] if filtered else [item[1] for item in ranked[:2]]

# FORMAT CITATIONS
def format_citations(context_list):
    return [f"{metadata[c['id']]['file']} - Page {metadata[c['id']]['page']}" for c in context_list]

# ASK THE MODEL (LEVEL-2 ANTI-HALLUCINATION)
def ask(query, debug=False):

    blocked_keywords = ["who", "invented", "inventor", "founder", "creator", "designed", "built by", "made by"]
    q_lower = query.lower()

    if any(word in q_lower for word in blocked_keywords):
        corpus = " ".join(chunks).lower()
        if not any(token in corpus for token in q_lower.split()):
            return {
                "answer": "This information is not available in the provided document(s).",
                "citations": [],
                "context_used": None
            }
    top_contexts = hybrid_retrieve(query, k=3)
    context_text = "\n\n".join(c["text"] for c in top_contexts)

    prompt = f"""
You are an aviation expert.
Answer ONLY using the context provided.

PRIORITY ORDER:
1. If a definition exists → provide ONLY the definition.
2. If examples/numbers exist but no definition → infer the definition (NO numbers).
   Format: INFERRED: <definition>
3. Never answer using numeric METAR examples for conceptual questions.
4. If answer cannot be found or confidently inferred → respond ONLY:
   "This information is not available in the provided document(s)."

NAMED ENTITY RESTRICTION:
- If the question asks about a person, inventor, founder, author, or any named individual,
  and the context does NOT explicit contain the answer, respond ONLY:
  "This information is not available in the provided document(s)."
- Do NOT use INFERRED for names or inventors.

STRICT RESTRICTIONS:
- No external knowledge or assumptions.
- No hallucinations.
- No invented facts.
- Do not reference missing context.
- Do not explain refusal.

------------------ CONTEXT ------------------
{context_text}
----------------------------------------------

Question: {query}
Answer:
"""

    response = ollama.chat(
        model="llama2:7b",
        messages=[
            {"role": "system", "content": "Use ONLY the provided context. No external knowledge."},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.0, "top_k": 20, "num_ctx": 1024}
    )

    answer = response["message"]["content"].strip()

    return {
        "answer": answer,
        "citations": format_citations(top_contexts),
        "context_used": context_text if debug else None
    }

if __name__ == "__main__":
    while True:
        q = input("\nAsk: ")
        print(ask(q, debug=True))
