from __future__ import annotations
import re
from typing import List

def normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def noun_phrase_candidates(text: str) -> List[str]:
    # Very light NP candidate heuristic (for QFS seed)
    # You can replace with spaCy noun_chunks if available.
    toks = re.findall(r"[A-Za-z0-9_\-/]+", text)
    out: List[str] = []
    buf: List[str] = []
    for t in toks:
        if t[0].isupper() or len(t) > 5:
            buf.append(t)
        else:
            if buf:
                out.append(" ".join(buf))
                buf = []
    if buf:
        out.append(" ".join(buf))
    return list(dict.fromkeys(out))[:10]
