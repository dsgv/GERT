"""Local SGPT-1.3B embedder exposing the OpenAIEmbedder interface.

Lets ``BaseSchemaLinkingRetrieverPPR`` and the Neo4j vector index use the same
SGPT-1.3B sentence encoder that backs the SingleDPR (SGPT) baseline, so GERT
can be rerun with the encoder swapped from BGE-M3 to SGPT-1.3B while every
other component (retrieval query, ranking, PPR, fusion) stays unchanged.
"""

from __future__ import annotations

from typing import Dict


class SGPTEmbedder:
    """Drop-in replacement for ``core.pipeline.retrieval.OpenAIEmbedder``."""

    def __init__(self, model_path: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_path, device=device)
        self.device = device
        # Usage counters kept for interface parity; local inference has no billing.
        self.total_tokens = 0
        self.total_requests = 0

    def embed_query(self, text: str):
        vector = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        self.total_requests += 1
        self.total_tokens += len(str(text).split())
        return vector.tolist()

    def get_token_usage(self) -> Dict[str, int]:
        return {"total_tokens": self.total_tokens, "total_requests": self.total_requests}

    def reset_token_usage(self) -> None:
        self.total_tokens = 0
        self.total_requests = 0
