from flask import Flask, request, jsonify, render_template
import json
import numpy as np
import faiss
from query_engine import search

from sentence_transformers import SentenceTransformer

app = Flask(__name__, template_folder="../templates")

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAISS_INDEX_PATH = os.path.join(BASE_DIR, "../data/processed/faiss.index")
META_PATH = os.path.join(BASE_DIR, "../data/processed/metadata.json")

# FAISS_INDEX_PATH = "../data/processed/faiss.index"
# META_PATH = "../data/processed/metadata.json"

# Load once (important)
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(FAISS_INDEX_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)


# def search(query, top_k=3):
#     query_vector = model.encode([query])
#     query_vector = np.array(query_vector)

#     distances, indices = index.search(query_vector, top_k)

#     results = []
#     for i in range(top_k):
#         idx = indices[0][i]
#         results.append({
#             "text": metadata[idx]["text"],
#             "title": metadata[idx]["title"],
#             "url": metadata[idx]["url"]
#         })

#     return results


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search_api():
    data = request.json
    query = data.get("query", "")

    # results = search(query)
    results = search(query, model, index, metadata)

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)