"""Offline script to build the vector index for the QA chatbot.

It performs three main steps:
1. Load documents from `data/` (both .txt and .pdf)
2. Split each document into overlapping chunks
3. Encode all chunks with a SentenceTransformer model and save:
   - `embeddings/vectors.npy`  (embeddings matrix)
   - `embeddings/metadata.json` (chunk ids, sources, and texts)

The CLI (`qa_cli.py`) later uses these artifacts for fast retrieval.
"""

import os
import json
import warnings
from pathlib import Path
from typing import List, Dict

import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import logging as hf_logging

# Silence detailed HF/transformers logging and specific hub warnings.
hf_logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="huggingface_hub.file_download",
)


DATA_DIR = Path("data")
EMB_DIR = Path("embeddings")
EMB_DIR.mkdir(exist_ok=True, parents=True)

# Simple character-based chunking configuration.
CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 100   # overlapping characters between chunks


def load_documents(data_dir: Path) -> List[Dict]:
    """Load both .txt and .pdf files from the data directory."""
    docs: List[Dict] = []

    # Load .txt files as plain text.
    for file in data_dir.glob("*.txt"):
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        docs.append({"path": str(file), "text": text})

    # Load .pdf files by extracting text from each page.
    for file in data_dir.glob("*.pdf"):
        try:
            reader = PdfReader(str(file))
            pages_text = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                pages_text.append(page_text)
            full_text = "\n".join(pages_text).strip()
            if full_text:
                docs.append({"path": str(file), "text": full_text})
        except Exception as e:
            print(f"Warning: Failed to read PDF {file}: {e}")

    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split a long text into overlapping character-based chunks.

    Overlap helps preserve context across neighboring chunks so that
    retrieval is less sensitive to exact chunk boundaries.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return [c for c in chunks if c]


def build_corpus(docs: List[Dict]) -> List[Dict]:
    """Convert raw documents into a flat list of chunk records."""
    corpus = []
    for doc in docs:
        chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, chunk in enumerate(chunks):
            corpus.append(
                {
                    "id": f"{doc['path']}::chunk_{idx}",
                    "source": doc["path"],
                    "text": chunk,
                }
            )
    return corpus


def main():
    """Entry point for building the embeddings index."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist.")

    print(f"Loading documents from {DATA_DIR}...")
    docs = load_documents(DATA_DIR)
    if not docs:
        raise ValueError("No .txt files found in data directory. Add some documents first.")

    print(f"Chunking {len(docs)} document(s)...")
    corpus = build_corpus(docs)
    texts = [c["text"] for c in corpus]
    print(f"Total chunks: {len(texts)}")

    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding chunks...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

    print("Saving embeddings and metadata...")
    np.save(EMB_DIR / "vectors.npy", embeddings)

    meta_path = EMB_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"Saved {embeddings.shape[0]} vectors to {EMB_DIR}")
    print("Index build complete.")


if __name__ == "__main__":
    main()

