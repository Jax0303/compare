from __future__ import annotations
import os, json, hashlib, threading
from typing import Optional, List

_lock = threading.Lock()

def _default_path() -> str:
    return os.getenv("EMBED_CACHE_PATH", "runs/embed_cache.jsonl")

def _key_for(text: str, dim: int, model: str) -> str:
    h = hashlib.sha1((text or "").encode("utf-8")).hexdigest()
    return f"{model}:{dim}:{h}"

def get_cached_embedding(text: str, dim: int, model: str, path: Optional[str] = None) -> Optional[List[float]]:
    path = path or _default_path()
    if not os.path.exists(path):
        return None
    k = _key_for(text, dim, model)
    try:
        with _lock, open(path, "r", encoding="utf-8") as f:
            for ln in f:
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                if rec.get("key") == k:
                    v = rec.get("vec")
                    if isinstance(v, list) and len(v) == dim:
                        return [float(x) for x in v]
    except Exception:
        return None
    return None

def put_cached_embedding(text: str, dim: int, model: str, vec: List[float], path: Optional[str] = None) -> None:
    path = path or _default_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rec = {"key": _key_for(text, dim, model), "vec": [float(x) for x in vec]}
    try:
        with _lock, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


