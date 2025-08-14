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
    # 신규 SDK에서 권장되는 Safety 설정 (필요 시 완화)
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUAL_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        gen_cfg = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            candidate_count=1,
        )
    except Exception:
        safety_settings, gen_cfg = None, {"temperature": temperature, "max_output_tokens": max_output_tokens}

    model = genai.GenerativeModel(MODEL_NAME if MODEL_NAME.startswith("models/") else f"models/{MODEL_NAME}")

    def _extract_text(resp):
        # 1순위: resp.text
        try:
            if getattr(resp, "text", None):
                return resp.text.strip()
        except Exception:
            pass
        # 2순위: candidates.parts[].text
        out = []
        for c in getattr(resp, "candidates", []) or []:
            content = getattr(c, "content", None)
            parts = getattr(content, "parts", []) if content is not None else []
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    out.append(t)
        return "\n".join(out).strip()

    try:
        resp = model.generate_content(
            contents=[{"role": "user", "parts": [{"text": (prompt or '').strip()}]}],
            generation_config=gen_cfg,
            safety_settings=safety_settings,
        )
        text = _extract_text(resp)
        if text:
            return text

        # 텍스트가 비면 이유를 표면화
        pf = getattr(resp, "prompt_feedback", None)
        sr = [getattr(c, "safety_ratings", None) for c in getattr(resp, "candidates", []) or []]
        debug = f"[empty_response] prompt_feedback={pf}, safety_ratings={sr}"
        # 마지막 시도: flash로 폴백
        try:
            fallback = genai.GenerativeModel("models/gemini-2.0-flash")
            resp2 = fallback.generate_content(
                contents=[{"role": "user", "parts": [{"text": (prompt or '').strip()}]}],
                generation_config=gen_cfg,
                safety_settings=safety_settings,
            )
            text2 = _extract_text(resp2)
            return text2 if text2 else debug
        except Exception:
            return debug
    except Exception as e:
        return f"[generate_error] {type(e).__name__}: {e}"

