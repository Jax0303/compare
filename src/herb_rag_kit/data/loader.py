from __future__ import annotations
import json, os
from typing import Dict, Any, Iterable, List

def iter_questions(path_jsonl: str) -> Iterable[Dict[str, Any]]:
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def iter_corpus(corpus_dir: str) -> Iterable[Dict[str, Any]]:
    for fn in os.listdir(corpus_dir):
        if not fn.endswith('.json'):
            continue
        p = os.path.join(corpus_dir, fn)
        with open(p, 'r', encoding='utf-8') as f:
            try:
                doc = json.load(f)
                yield doc
            except Exception:
                continue
