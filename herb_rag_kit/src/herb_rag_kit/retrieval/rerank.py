from __future__ import annotations
from typing import List, Tuple
from rapidfuzz.distance import Levenshtein

def simple_rerank(query: str, docs: List[Tuple[str, float, str]], top_k: int = 10) -> List[Tuple[str, float, str]]:
    rescored: List[Tuple[str, float, str]] = []
    q = (query or "")
    if not isinstance(q, str): q = str(q)
    q = q[:256]
    for doc_id, score, text in docs:
        t = (text or "")
        if not isinstance(t, str): t = str(t)
        t = t[:256]
        try:
            dist = Levenshtein.distance(q, t)
        except Exception:
            dist = 0
        rescored.append((doc_id, score - 0.001*dist, text))
    return sorted(rescored, key=lambda x: x[1], reverse=True)[:top_k]
