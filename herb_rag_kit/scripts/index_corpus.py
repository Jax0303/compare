#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, glob, json, argparse, hashlib, pickle
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
import numpy as np

# ------- 파싱 / 토크나이즈 -------
TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣_]+")
def tokenize(s: str) -> List[str]:
    return TOKEN_RE.findall((s or "").lower())

def iter_corpus(corpus_dir: str):
    for p in glob.glob(os.path.join(corpus_dir, "*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        text = (d.get("text") or d.get("content") or d.get("message") or "").strip()
        if not text:
            continue
        doc_id = str(d.get("id") or os.path.basename(p))
        yield {"id": doc_id, "text": text}

def content_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

# ------- 임베딩 -------
def st_embed(texts: List[str], batch_size: int = 256) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="embedding(st)"):
        chunk = texts[i:i+batch_size]
        v = model.encode(chunk, normalize_embeddings=True, convert_to_numpy=True)
        vecs.append(v.astype("float32"))
    return np.vstack(vecs) if vecs else np.empty((0, 384), dtype="float32")

# ------- 인덱싱 엔트리 -------
def build_bm25(texts: List[str]):
    from rank_bm25 import BM25Okapi
    toks = [tokenize(t) for t in texts]
    return BM25Okapi(toks), toks

def build_faiss(vecs: np.ndarray):
    import faiss  # type: ignore
    d = int(vecs.shape[1])
    index = faiss.IndexFlatIP(d)
    index.add(vecs)
    return index

def save_jsonl(path: Path, records: List[Dict]):
    with open(path, "w", encoding="utf-8") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", required=True, help="HERB/data/corpus")
    ap.add_argument("--index_dir", default="indexes/herb", help="인덱스 출력 디렉토리")
    ap.add_argument("--rebuild", action="store_true", help="캐시 무시하고 강재생성")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # 1) 파싱
    docs = list(iter_corpus(args.corpus_dir))
    if not docs: raise SystemExit(f"No docs under {args.corpus_dir}")
    ids  = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    hashes = [content_hash(t) for t in texts]

    # 2) 이전 스냅샷과 비교(캐시)
    ids_path    = index_dir / "ids.json"
    hash_path   = index_dir / "hashes.json"
    bm25_path   = index_dir / "bm25.pkl"
    toks_path   = index_dir / "bm25_tokens.pkl"
    vecs_path   = index_dir / "embeddings.npy"
    faiss_path  = index_dir / "faiss.index"
    store_path  = index_dir / "docstore.jsonl"
    meta_path   = index_dir / "meta.json"

    prev_ids = json.load(open(ids_path, "r", encoding="utf-8")) if ids_path.exists() else None
    prev_hash= json.load(open(hash_path,"r", encoding="utf-8")) if hash_path.exists() else None

    same_corpus = (prev_ids == ids and prev_hash == hashes)

    # 3) 도큐먼트 스토어 기록
    save_jsonl(store_path, docs)
    json.dump(ids,   open(ids_path, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(hashes,open(hash_path,"w", encoding="utf-8"), ensure_ascii=False)

    # 4) BM25 (토큰 캐시 + 인덱스)
    if args.rebuild or (not same_corpus) or (not bm25_path.exists()):
        bm25, toks = build_bm25(texts)
        with open(bm25_path, "wb") as w: pickle.dump(bm25, w)
        with open(toks_path, "wb") as w: pickle.dump(toks, w)
    else:
        print("BM25 unchanged → reuse")

    # 5) Dense 임베딩 + FAISS
    need_dense = args.rebuild or (not same_corpus) or (not faiss_path.exists()) or (not vecs_path.exists())
    if need_dense:
        vecs = st_embed(texts)
        np.save(vecs_path, vecs)
        index = build_faiss(vecs)
        import faiss
        faiss.write_index(index, str(faiss_path))
    else:
        print("Dense unchanged → reuse")

    # 6) 메타
    meta = {
        "size": len(docs),
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "created": __import__("datetime").datetime.now().isoformat(),
    }
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"✅ Indexed: {len(docs)} docs → {index_dir}")

if __name__ == "__main__":
    main()
