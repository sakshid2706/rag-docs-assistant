# rag-docs-assistant
An end-to-end Retrieval-Augmented Generation (RAG) system for semantic document search, built from scratch using FAISS, Sentence Transformers, and Flask — with a focus on clean pipeline design, retrieval quality, and explainability.

# Tech Stack
### Backend
* Python
* Flask

### Core ML/NLP
* Sentence Transformers (semantic embeddings)
* FAISS (vector similarity search)

### Data Processing
* BeautifulSoup
* Requests
* JSON

### Testing
* pytest
* pytest-cov

# Pipeline Breakdown
## 1. scraper.py
* Fetches documentation pages using Requests
* Uses BeautifulSoup to parse HTML
* Extracts main content and removes noise

## 2. cleaner.py
* Removes short/noisy lines
* Cleans text formatting
* Extracts document titles

## 3. chunker.py
* Splits documents into overlapping chunks
* Preserves context for better retrieval

## 4. embedder.py
* Converts chunks into embeddings using Sentence Transformers
* Stores vectors in FAISS index

## 5. query_engine.py
* Converts query into embedding
* Searches FAISS index
* Returns top relevant chunks
* Enhancements:
- Deduplication (removes repeated chunks)
- Diversity (ensures varied results)

## 6. app.py
* Flask API for querying
* Connects frontend with retrieval pipeline

### Example
<img width="2936" height="1266" alt="image" src="https://github.com/user-attachments/assets/1384a822-682e-4c95-8c16-0e59f4a132fa" />




