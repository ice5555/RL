# retrieval/rerank_ce.py
from __future__ import annotations
import numpy as np
import torch
from sentence_transformers import CrossEncoder

def _auto_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str | None = None, batch_size: int = 32):
        self.device = device or _auto_device()
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, docs: list[str], row_ids: list[int], top_k: int):
        if not docs:
            return row_ids[:top_k]
        pairs = [[query, d] for d in docs]
        scores = np.asarray(self.model.predict(pairs, batch_size=self.batch_size)).flatten()
        order = np.argsort(scores)[::-1][: min(top_k, len(row_ids))]
        return [row_ids[i] for i in order]