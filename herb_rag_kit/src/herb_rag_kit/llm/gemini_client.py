from __future__ import annotations
from typing import List
import os
from pathlib import Path
import google.generativeai as genai
from ..config import SETTINGS

MODEL_NAME = "gemini-2.5-pro"
EMBED_MODEL = "text-embedding-004"

def ensure_configured():
    # 루트 .env도 시도 (현재 cwd 달라도 읽히게)
    if not SETTINGS.gemini_api_key:
        env_guess = Path(__file__).resolve().parents[2] / ".env"
        if env_guess.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_guess, override=False)
            except Exception:
                pass
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Put it in repo_root/.env or export it.")
    genai.configure(api_key=api_key)

def _unwrap_vec(e):
    # v0.7+ 스타일: {"embedding":{"values":[...]}}
    if isinstance(e, dict):
        if "embedding" in e and isinstance(e["embedding"], dict) and "values" in e["embedding"]:
            return e["embedding"]["values"]
        if "values" in e:
            return e["values"]
    return e  # 이미 list 면 그대로

def embed_texts(texts: List[str]) -> List[List[float]]:
    ensure_configured()
    safe = [(t or "")[:4000] for t in texts]
    embs: List[List[float]] = []
    B = 128
    for i in range(0, len(safe), B):
        chunk = safe[i:i+B]
        out = genai.embed_content(model=EMBED_MODEL, content=chunk, task_type="retrieval_document")
        batch = []
        if isinstance(out, dict) and "embeddings" in out:
            for e in out["embeddings"]:
                v = _unwrap_vec(e)
                batch.append(v if isinstance(v, list) else [0.0]*768)
        elif isinstance(out, dict) and "embedding" in out:
            v = _unwrap_vec(out["embedding"])
            batch = [v if isinstance(v, list) else [0.0]*768]
        else:
            batch = [[0.0]*768 for _ in chunk]
        embs.extend(batch)
    return embs

def embed_query(text: str) -> List[float]:
    ensure_configured()
    out = genai.embed_content(model=EMBED_MODEL, content=(text or "")[:4000], task_type="retrieval_query")
    if isinstance(out, dict) and "embedding" in out:
        v = _unwrap_vec(out["embedding"])
        return v if isinstance(v, list) else [0.0]*768
    if isinstance(out, dict) and "embeddings" in out:
        v = _unwrap_vec(out["embeddings"][0])
        return v if isinstance(v, list) else [0.0]*768
    return [0.0]*768

def generate(prompt: str, temperature: float = 0.2, max_output_tokens: int = 512) -> str:
    ensure_configured()
    model = genai.GenerativeModel(MODEL_NAME)
    try:
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens},
        )
    except Exception:
        return ""
    # 보통 케이스
    try:
        if resp.text:
            return resp.text
    except Exception:
        pass
    # 후보 파트에서 텍스트 수집(세이프티/빈 후보 대응)
    out_parts = []
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", []) if content is not None else []
        for p in parts:
            t = getattr(p, "text", None)
            if t:
                out_parts.append(t)
    return "\n".join(out_parts).strip()
