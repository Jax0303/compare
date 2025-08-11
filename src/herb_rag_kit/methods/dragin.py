from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
from ..utils.text import noun_phrase_candidates
from ..llm.gemini_client import generate

def self_consistency_variance(prompt: str, n: int = 3) -> float:
    # Diversity as proxy for uncertainty
    outs = [generate(prompt, temperature=1.0, max_output_tokens=64) for _ in range(n)]
    # Very light bag-of-words variance
    import re, collections
    vecs = []
    vocab = {}
    for o in outs:
        toks = re.findall(r"[A-Za-z0-9_\-]+", o.lower())
        cnt = collections.Counter(toks)
        vecs.append(cnt)
        for t in cnt:
            vocab[t] = True
    vocab = list(vocab.keys())
    mat = []
    for cnt in vecs:
        mat.append([cnt.get(t,0) for t in vocab])
    x = np.array(mat, dtype=float)
    if x.size == 0:
        return 0.0
    v = float(np.var(x))
    return v

def rind_should_retrieve(question: str) -> bool:
    # Lightweight RIND: ask classifier + diversity check
    probe = (
        "Answer briefly. Do you need external factual information to answer this question accurately? "
        "Reply with YES or NO only. Question: " + question
    )
    resp = generate(probe, temperature=0.0, max_output_tokens=8).strip().upper()
    diversity = self_consistency_variance("Give a short factual answer: " + question, n=3)
    return ("YES" in resp) or (diversity > 0.2)

def qfs_build_query(question: str, context_hint: str | None = None) -> str:
    seeds = noun_phrase_candidates(question)
    base = "; ".join(seeds[:5])
    if context_hint:
        base += " | " + context_hint
    refine = generate(f"Refine this into a search query: {question}\nSeeds: {base}\nQuery:", temperature=0.2, max_output_tokens=32)
    return refine.strip() or (question + " " + base)
