import os
import json
import faiss
import pickle
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# FOLDER PATHS
DOC_FOLDER = r"C:\Users\sujit\Desktop\armiango\data\pdfs"
INDEX_PATH = "vector_store/index.faiss"
CHUNKS_PATH = "vector_store/chunks.pkl"
META_PATH = "app/metadata.json"

# Embedding Model
EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


# PDF LOADING + CHUNKING 
def load_and_chunk(pdf_path, chunk_size=350, overlap=50):
    """Load PDF, extract text page-by-page + chunk"""
    reader = PdfReader(pdf_path)
    chunks = []
    metadata = []

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


# INGEST PIPELINE 
def ingest():
    all_chunks = []
    all_meta = []

    print(f"\n Scanning folder: {DOC_FOLDER}")
    print(" Starting ingestion...\n")

    os.makedirs("vector_store", exist_ok=True)
    os.makedirs("app", exist_ok=True)

    for file in os.listdir(DOC_FOLDER):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(DOC_FOLDER, file)
            print(f" Processing: {file}")

            chunks, metadata = load_and_chunk(pdf_path)
            all_chunks.extend(chunks)
            all_meta.extend(metadata)

    if len(all_chunks) == 0:
        print(" No PDFs found in /documents. Add files and try again.")
        return

    print(f"\n Total Chunks Created: {len(all_chunks)}")
    print(" Generating embeddings... This may take a while for big PDFs...\n")

    vectors = EMB_MODEL.encode(all_chunks, batch_size=64, show_progress_bar=True)

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors, dtype="float32"))

    faiss.write_index(index, INDEX_PATH)
    pickle.dump(all_chunks, open(CHUNKS_PATH, "wb"))
    json.dump(all_meta, open(META_PATH, "w"))

    print("\n INGESTION COMPLETED SUCCESSFULLY!")
    print(f" Vector Index Saved to: {INDEX_PATH}")
    print(f" Chunks Saved to: {CHUNKS_PATH}")
    print(f" Metadata Saved to: {META_PATH}")




if __name__ == "__main__":
    ingest()
