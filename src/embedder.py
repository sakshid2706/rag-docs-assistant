import json
import os
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

INPUT_PATH = "../data/processed/chunks.json"
EMBEDDING_PATH = "../data/processed/embeddings.npy"
META_PATH = "../data/processed/metadata.json"
FAISS_INDEX_PATH = "../data/processed/faiss.index"


def load_chunks():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def create_embeddings(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    return np.array(embeddings)


def save_embeddings(embeddings):
    np.save(EMBEDDING_PATH, embeddings)


def save_metadata(chunks):
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)   # simple & effective
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"FAISS index built with {index.ntotal} vectors.")


def main():
    os.makedirs("../data/processed", exist_ok=True)

    # Load chunks
    chunks = load_chunks()

    if len(chunks) == 0:
        print("No chunks found. Fix previous steps.")
        return

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create embeddings
    embeddings = create_embeddings(chunks, model)

    # Save everything
    save_embeddings(embeddings)
    save_metadata(chunks)
    build_faiss_index(embeddings)

    print("Embedding pipeline complete!")


if __name__ == "__main__":
    main()