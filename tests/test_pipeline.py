import sys
import os

sys.path.append(os.path.abspath("src"))

from cleaner import clean_text_v2
from chunker import split_into_chunks
from query_engine import search, load_index, load_metadata

from sentence_transformers import SentenceTransformer


def test_clean_text_removes_noise():
    text = "Hi\n\nCreated On: test\nThis is a meaningful sentence with enough words."

    cleaned = clean_text_v2(text)

    assert "Created On" not in cleaned
    assert len(cleaned) > 0


def test_chunking_creates_multiple_chunks():
    text = "word " * 500
    chunks = split_into_chunks(text)

    assert len(chunks) > 1
    assert isinstance(chunks[0], str)


def test_retrieval_returns_results():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = load_index()
    metadata = load_metadata()

    results = search("tensor", model, index, metadata)

    assert len(results) > 0
    assert "text" in results[0]
    assert "title" in results[0]
    assert "url" in results[0]

def test_retrieval_diversity():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = load_index()
    metadata = load_metadata()

    results = search("tensor", model, index, metadata)

    urls = [r["url"] for r in results]

    assert len(set(urls)) == len(urls)

def test_api_search():
    from app import app

    client = app.test_client()

    response = client.post("/search", json={"query": "tensor"})

    assert response.status_code == 200
    assert isinstance(response.json, list)

def test_extract_title():
    text = "This is a proper title\nSome content below"

    from cleaner import extract_title

    title = extract_title(text)

    assert title == "This is a proper title"

def test_embedding_generation():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embedding = model.encode(["test sentence"])

    assert len(embedding[0]) > 0

def test_embedding_runs():
    from embedder import create_embeddings
    from sentence_transformers import SentenceTransformer

    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Minimal dummy input
    chunks = [
        {"text": "This is a test chunk", "title": "Test", "url": "test_url"}
    ]

    embeddings = create_embeddings(chunks, model)

    assert len(embeddings) > 0