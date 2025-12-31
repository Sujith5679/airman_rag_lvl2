import os
import json
import faiss
import pickle
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# FOLDER PATHS
DOC_FOLDER = r"C:\Users\sujit\Desktop\airmanc\data\pdfs"
INDEX_PATH = "vector_store/index.faiss"
CHUNKS_PATH = "vector_store/chunks.pkl"
META_PATH = "app/metadata.json"

# MODEL
EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# PDF LOADING + DYNAMIC CHUNKING
def load_and_chunk(pdf_path, chunk_size=250, overlap=100):
    """Load PDF, extract text page-by-page and chunk into text segments"""
    reader = PdfReader(pdf_path)
    chunks, metadata = [], []

    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except:
            text = None

        if not text:
            continue

        words = text.split()

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            metadata.append({
                "file": os.path.basename(pdf_path),
                "page": page_num + 1,
                "chunk_index": len(chunks) - 1
            })

    return chunks, metadata

# MAIN INGEST PIPELINE
def ingest():
    all_chunks = []
    all_meta = []

    print(f"\n Scanning folder: {DOC_FOLDER}")
    os.makedirs("vector_store", exist_ok=True)
    os.makedirs("app", exist_ok=True)

    for file in os.listdir(DOC_FOLDER):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(DOC_FOLDER, file)
            file_lower = file.lower()

            print(f"\n Processing: {file}")

            if "meteorology" in file_lower or "turbulence" in file_lower:
                print(" Using meteorology settings (350/150)")
                chunks, metadata = load_and_chunk(pdf_path, chunk_size=350, overlap=150)

            elif "regulation" in file_lower or "rules" in file_lower or "air" in file_lower:
                print(" Using regulation settings (300/120)")
                chunks, metadata = load_and_chunk(pdf_path, chunk_size=300, overlap=120)

            else:
                print(" Using default settings (250/100)")
                chunks, metadata = load_and_chunk(pdf_path, chunk_size=250, overlap=100)

            all_chunks.extend(chunks)
            all_meta.extend(metadata)

    if not all_chunks:
        print(" No PDF files found. Add files in /pdfs and try again.")
        return

    print(f"\n Total Chunks Created: {len(all_chunks)}")
    print(" Generating vector embeddings...\n")

    vectors = EMB_MODEL.encode(all_chunks, batch_size=64, show_progress_bar=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors, dtype="float32"))

    faiss.write_index(index, INDEX_PATH)
    pickle.dump(all_chunks, open(CHUNKS_PATH, "wb"))
    json.dump(all_meta, open(META_PATH, "w"))

    print("\n Building BM25 index...")
    bm25_corpus = [chunk.split() for chunk in all_chunks]
    bm25 = BM25Okapi(bm25_corpus)

    with open("vector_store/bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)

    print("\n INGESTION COMPLETE!")
    print(f" FAISS Index  → {INDEX_PATH}")
    print(f" Chunk Store  → {CHUNKS_PATH}")
    print(f" Metadata     → {META_PATH}")
    print(f" BM25 Index   → vector_store/bm25_index.pkl\n")
    
if __name__ == "__main__":
    ingest()
