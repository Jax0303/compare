# src/herb_rag_kit/data/loader.py
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple

# -------------------------
# 기본 유틸 (질문/코퍼스 리더)
# -------------------------
def iter_questions(path_jsonl: str) -> Iterable[Dict[str, Any]]:
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def iter_corpus(corpus_dir: str) -> Iterable[Dict[str, Any]]:
    for fn in os.listdir(corpus_dir):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(corpus_dir, fn)
        try:
            with open(p, "r", encoding="utf-8") as f:
                yield json.load(f)
        except Exception:
            continue

# ----------------------------------------
# HERB products-aware loader (핵심 패치)
# ----------------------------------------
_HERB_EXCLUDE_KEYS = {"team", "customers"}  # RAG 평가 제외 권고
_TEXT_KEYS = [
    "text", "message", "content", "body", "description", "title", "summary",
    "transcript", "note", "raw", "snippet"
]

def _pick_text(obj: Any) -> str:
    """dict/list/str 어디서든 문자열 잎사귀를 수집해 합침(불필요한 키 제외)"""
    out: List[str] = []

    def rec(x: Any, parent_key: str | None = None):
        if x is None:
            return
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
            return
        if isinstance(x, dict):
            for k, v in x.items():
                if k in _HERB_EXCLUDE_KEYS:
                    continue
                # text 계열 키는 우선 처리
                if isinstance(v, str) and k in _TEXT_KEYS:
                    sv = v.strip()
                    if sv:
                        out.append(sv)
                    continue
                rec(v, k)
            return
        if isinstance(x, list):
            for it in x:
                rec(it, parent_key)
            return
        # 숫자/기타 타입은 무시

    rec(obj)
    return "\n".join(out)[:20000]  # 너무 길어지지 않게 상한

def _iter_product_docs(products_dir: Path) -> Iterable[Tuple[str, str]]:
    """products/*.json을 펼쳐 (doc_id, text) 생성 — 아티팩트 단위 문서화"""
    for p in sorted(products_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        product = p.stem
        artifact_keys = [
            "slack", "documents", "docs", "meeting_transcripts", "meeting_chats",
            "urls", "pull_requests", "prs", "artifacts", "issues", "tickets",
            "notes"
        ]
        found = False
        for key in artifact_keys:
            items = data.get(key)
            if not isinstance(items, list):
                continue
            found = True
            for idx, item in enumerate(items):
                text = None
                if isinstance(item, dict):
                    parts: List[str] = []
                    for kk in ["title", "message", "content", "body", "text", "transcript", "description", "summary"]:
                        vv = item.get(kk)
                        if isinstance(vv, str) and vv.strip():
                            parts.append(vv.strip())
                    text = "\n".join(parts) if parts else _pick_text(item)
                elif isinstance(item, str):
                    text = item.strip()

                if text and text.strip():
                    doc_id = f"{product}/{key}/{item.get('id', idx) if isinstance(item, dict) else idx}"
                    yield doc_id, text

        # 아티팩트 키를 전혀 못 찾았으면 전체 dict에서 문자열만 긁어 하나의 문서로라도 생성
        if not found:
            text = _pick_text(data)
            if text and text.strip():
                yield f"{product}/full", text

def load_corpus(herb_root: str) -> Tuple[List[str], List[str]]:
    """
    HERB 전용: products가 있으면 products를 '우선' 사용(심볼릭 포함),
    없을 때만 corpus 폴더의 json을 문서로 간주.
    반환: (doc_ids, texts)
    """
    root = Path(herb_root).expanduser().resolve()
    corpus_dir = root / "data" / "corpus"
    products_dir = root / "data" / "products"

    doc_ids: List[str] = []
    texts: List[str] = []

    # 1) products 우선
    if products_dir.exists():
        for did, txt in _iter_product_docs(products_dir):
            if txt and txt.strip():
                doc_ids.append(did)
                texts.append(txt)
        if doc_ids:
            return doc_ids, texts

    # 2) fallback: corpus의 각 json을 하나의 문서로 처리
    if corpus_dir.exists():
        for p in sorted(corpus_dir.rglob("*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            text = None
            for k in _TEXT_KEYS:
                v = d.get(k)
                if isinstance(v, str) and v.strip():
                    text = v.strip()
                    break
            if not text:
                text = _pick_text(d)
            if text and text.strip():
                doc_ids.append(p.stem)
                texts.append(text)
        if doc_ids:
            return doc_ids, texts

    raise FileNotFoundError(f"No docs found under {corpus_dir} or {products_dir}")
