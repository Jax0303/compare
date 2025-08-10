from __future__ import annotations
from typing import List, Dict, Any
import os
import google.generativeai as genai
from ..config import SETTINGS

MODEL_NAME = "gemini-2.5-pro"
EMBED_MODEL = "text-embedding-004"

def ensure_configured():
    if not SETTINGS.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Please put it in environment or .env")
    genai.configure(api_key=SETTINGS.gemini_api_key)

def embed_texts(texts: List[str]) -> List[List[float]]:
    ensure_configured()
    out = genai.embed_content(model=EMBED_MODEL, content=texts, task_type="retrieval_document")
    # API returns different shapes depending on single/batch; normalize
    if isinstance(out, dict) and "embedding" in out:
        return [out["embedding"]]
    if isinstance(out, dict) and "embeddings" in out:
        return [e["values"] if isinstance(e, dict) and "values" in e else e for e in out["embeddings"]]
    raise RuntimeError("Unexpected embedding API response")

def embed_query(text: str) -> List[float]:
    ensure_configured()
    out = genai.embed_content(model=EMBED_MODEL, content=text, task_type="retrieval_query")
    if isinstance(out, dict) and "embedding" in out:
        return out["embedding"]
    if isinstance(out, dict) and "embeddings" in out:
        return out["embeddings"][0]
    raise RuntimeError("Unexpected embedding API response")

def generate(prompt: str, temperature: float = 0.2, max_output_tokens: int = 512) -> str:
    ensure_configured()
    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(prompt, generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens})
    return resp.text or ""
