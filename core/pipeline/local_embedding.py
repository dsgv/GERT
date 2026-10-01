"""
Local sentence-transformers embeddings: shared by KG construction and retrieval.
"""
import os
from typing import List


def load_sentence_transformer(local_path: str):
    """Load SentenceTransformer model from disk."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Local embedding requires the 'sentence-transformers' package. "
            "Install it (e.g. pip install sentence-transformers) or use API embedding."
        ) from e
    path = os.path.abspath(local_path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Local embedding model path does not exist: {path}")
    return SentenceTransformer(path)


def embed_batch_sentence_transformer(st_model, texts: List[str]) -> List[List[float]]:
    """Batch encode, return Python float list (consistent with Neo4j / OpenAI vector format)."""
    import numpy as np

    arr = st_model.encode(
        texts,
        batch_size=len(texts),
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    if isinstance(arr, np.ndarray):
        return [row.tolist() for row in arr]
    return [list(row) for row in arr]
