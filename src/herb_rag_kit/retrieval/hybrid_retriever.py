from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np

class HybridRetriever:
    def __init__(self, bm25, emb_index, alpha: float = 0.5):
        self.bm25 = bm25
        self.emb_index = emb_index
        self.alpha = alpha

    def search(self, query: str, qvec: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        bm_hits = self.bm25.search(query, k=k*5)
        em_hits = self.emb_index.search(qvec, k=k*5)
        scores: Dict[str, float] = {}
        for i, (d, s) in enumerate(bm_hits):
            scores[d] = max(scores.get(d, 0.0), (1.0 - self.alpha) * s)
        for i, (d, s) in enumerate(em_hits):
            scores[d] = max(scores.get(d, 0.0), self.alpha * s + scores.get(d, 0.0))
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return ranked
