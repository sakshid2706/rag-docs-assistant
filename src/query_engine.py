import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
import os

# FAISS_INDEX_PATH = "../data/processed/faiss.index"
# META_PATH = "../data/processed/metadata.json"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAISS_INDEX_PATH = os.path.join(BASE_DIR, "../data/processed/faiss.index")
META_PATH = os.path.join(BASE_DIR, "../data/processed/metadata.json")


def load_index():
    return faiss.read_index(FAISS_INDEX_PATH)


def load_metadata():
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# def search(query, model, index, metadata, top_k=5):
#     query_vector = model.encode([query])
#     query_vector = np.array(query_vector)

#     distances, indices = index.search(query_vector, top_k)

#     seen = set()
#     results = []

#     for i in range(len(indices[0])):
#         idx = indices[0][i]
#         text = metadata[idx]["text"]

#         # Remove near-duplicate chunks
#         if text[:100] in seen:
#             continue

#         seen.add(text[:100])

#         results.append({
#             "score": float(distances[0][i]),
#             "text": text,
#             "title": metadata[idx]["title"],
#             "url": metadata[idx]["url"]
#         })

#         # limit final results
#         if len(results) == 3:
#             break

#     return results

def search(query, model, index, metadata, top_k=5):
    query_vector = model.encode([query])
    query_vector = np.array(query_vector)

    distances, indices = index.search(query_vector, top_k)

    seen_docs = set()
    results = []

    for i in range(len(indices[0])):
        idx = indices[0][i]
        doc_id = metadata[idx]["url"]   # using URL as doc identity

        # Ensure diversity: only one chunk per document
        if doc_id in seen_docs:
            continue

        seen_docs.add(doc_id)

        results.append({
            "score": float(distances[0][i]),
            "text": metadata[idx]["text"],
            "title": metadata[idx]["title"],
            "url": metadata[idx]["url"]
        })

        if len(results) == 3:
            break

    return results


def main():
    print("Loading model and index...")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = load_index()
    metadata = load_metadata()

    print("Ready! Ask your question (type 'exit' to quit)\n")

    while True:
        query = input(">>> ")

        if query.lower() == "exit":
            break

        results = search(query, model, index, metadata)

        print("\nTop Results:\n")
        for i, res in enumerate(results):
            print(f"Result {i+1}")
            print(f"Score: {res['score']:.4f}")
            print(f"Title: {res['title']}")
            print(f"URL: {res['url']}")
            print(f"Text: {res['text'][:300]}...")
            print("-" * 50)

        print("\n")


if __name__ == "__main__":
    main()