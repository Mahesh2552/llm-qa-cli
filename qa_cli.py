"""CLI retrieval-based QA assistant over local documents.

This script does NOT call a large generative LLM. Instead, it:
- loads pre-computed document chunk embeddings from `embeddings/`
- embeds the user question with the same model
- retrieves the most similar chunks
- inside those chunks, finds the single best-matching sentence

The final answer is that sentence (or "not in the context" if nothing matches),
so answers are always grounded in your local data.
"""

import json
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import logging as hf_logging

# Silence detailed HF/transformers logging and specific hub warnings.
hf_logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="huggingface_hub.file_download",
)


EMB_DIR = Path("embeddings")
# Number of most similar document chunks to inspect for answering a question.
TOP_K = 3


def load_index():
    """Load the dense vector index and corresponding chunk metadata.

    Returns
    -------
    vectors : np.ndarray
        2D array of shape (num_chunks, embedding_dim) with all chunk embeddings.
    metadata : list[dict]
        Metadata for each chunk, including `source` (file path) and `text`.
    """
    vectors_path = EMB_DIR / "vectors.npy"
    meta_path = EMB_DIR / "metadata.json"

    if not vectors_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Embeddings not found. Run 'python build_index.py' first to build the index."
        )

    vectors = np.load(vectors_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return vectors, metadata


def retrieve(
    query: str,
    emb_model: SentenceTransformer,
    vectors: np.ndarray,
    metadata: List[dict],
    top_k: int = TOP_K,
) -> List[Tuple[float, dict]]:
    # Embed the question once
    query_emb = emb_model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, vectors)[0]
    top_idx = np.argsort(-sims)[:top_k]

    results: List[Tuple[float, dict]] = []
    for idx in top_idx:
        results.append((float(sims[idx]), metadata[int(idx)]))
    return results


def extract_best_sentence(context_chunks: List[str], question: str, emb_model: SentenceTransformer) -> str:
    """Find the single sentence that best answers the question.

    Strategy:
    - split each retrieved chunk into lines/sentences
    - embed all sentences plus the question
    - pick the sentence with highest cosine similarity

    This is a simple non-generative QA approach that still uses embeddings,
    but avoids incorrect free-form generations.
    """
    # Collect candidate sentences (or individual lines) from all chunks.
    sentences = []
    for chunk in context_chunks:
        for line in chunk.split("\n"):
            s = line.strip()
            if s:
                sentences.append(s)

    if not sentences:
        return "not in the context"

    # Embed question and all candidate sentences.
    q_emb = emb_model.encode([question], convert_to_numpy=True)
    sent_embs = emb_model.encode(sentences, convert_to_numpy=True)
    sims = cosine_similarity(q_emb, sent_embs)[0]

    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    # If the best similarity is too low, treat as "no answer in context".
    if best_score < 0.2:
        return "not in the context"

    return sentences[best_idx]


def main():
    """Entry point for the CLI QA chatbot.

    Loads the index and embedding model once, then enters a REPL
    where the user types questions and receives answers drawn from
    local documents.
    """
    print("Loading embeddings index...")
    vectors, metadata = load_index()

    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("\nQA Chatbot ready. Type your question, or 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if question.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break

        if not question:
            continue

        results = retrieve(question, emb_model, vectors, metadata, top_k=TOP_K)
        context_chunks = [r[1]["text"] for r in results]

        # Instead of asking a generative model (which may pick a wrong sentence
        # from a long chunk), pick the best matching sentence directly.
        answer = extract_best_sentence(context_chunks, question, emb_model)

        print("\nBot:", answer)
        print("\nTop context snippet:")
        if context_chunks:
            print(context_chunks[0].strip())
        # print("\n--- Sources ---")
        # for score, meta in results:
        #     print(f"- {meta['source']} (score={score:.3f})")
        print("----------------\n")


if __name__ == "__main__":
    main()

