from __future__ import annotations
from typing import Dict, Any, List, Tuple
import re

BUCKET_KEYWORDS = {
    "code": ["function", "class", "compile", "python", "error", "stack", "trace"],
    "slack": ["channel", "slack", "dm", "thread", "message"],
    "docs": ["document", "policy", "guide", "manual", "spec", "design"],
    "pr": ["pull request", "PR", "merge", "commit", "repo"],
    "people": ["employee", "manager", "team", "hr", "recruit"],
}

def route_bucket(question: str, meta: Dict[str, Any] | None = None, buckets: List[str] | None = None) -> str:
    q = question.lower()
    buckets = buckets or list(BUCKET_KEYWORDS.keys())
    scores = {b:0 for b in buckets}
    for b in buckets:
        for kw in BUCKET_KEYWORDS.get(b, []):
            if re.search(r"\b" + re.escape(kw.lower()) + r"\b", q):
                scores[b]+=1
    # fallback
    best = max(scores.items(), key=lambda x:x[1])[0]
    return best

def hybrid_stage(bm25, emb_index, reranker, query: str, qvec, k: int = 10):
    bm = bm25.search(query, k=k*5)
    em = emb_index.search(qvec, k=k*5)
    # union with normalized scores
    sc = {}
    for d,s in bm: sc[d] = max(sc.get(d,0.0), 0.5*s)
    for d,s in em: sc[d] = max(sc.get(d,0.0), 0.5*s + sc.get(d,0.0))
    ranked = sorted(sc.items(), key=lambda x:x[1], reverse=True)[:k*2]
    # rerank with snippet distance
    # we expect caller to pass actual doc texts
    return ranked
