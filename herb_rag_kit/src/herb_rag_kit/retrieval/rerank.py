from __future__ import annotations
from typing import List, Tuple, Any, Dict
from rapidfuzz.distance import Levenshtein

def simple_rerank(query: str, docs: List[Tuple[str, float, str]], top_k: int = 10) -> List[Tuple[str, float, str]]:
    # Rerank by negative edit distance between query and snippet (toy but deterministic)
    rescored: List[Tuple[str, float, str]] = []
    for doc_id, score, text in docs:
        dist = Levenshtein.distance(query[:256], text[:256])
        rescored.append((doc_id, score - 0.001*dist, text))
    return sorted(rescored, key=lambda x: x[1], reverse=True)[:top_k]
