import json
import os

INPUT_PATH = "../data/processed/cleaned_docs.json"
OUTPUT_PATH = "../data/processed/chunks.json"

CHUNK_SIZE = 400        # words
OVERLAP = 100           # words


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

        start += (chunk_size - overlap)

    return chunks


def create_chunks():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    all_chunks = []

    for doc_id, doc in enumerate(docs):
        content = doc["content"]
        title = doc["title"]
        url = doc["url"]

        chunks = split_into_chunks(content)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc_id}_{i}",
                "text": chunk,
                "title": title,
                "url": url
            })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Chunking complete! {len(all_chunks)} chunks created.")


if __name__ == "__main__":
    create_chunks()