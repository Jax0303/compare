#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, glob, json, argparse
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣_]+")
def tok(s: str) -> List[str]:
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

def load_questions(path_jsonl: str) -> List[Dict]:
    if not path_jsonl: return []
    qs = []
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            qid = o.get("id") or o.get("_id") or o.get("qid")
            qtx = o.get("question") or o.get("query") or o.get("q")
            if qtx: qs.append({"id": qid, "question": qtx})
    return qs

def rrf(ranked: List[Tuple[str, float]], k: int = 60) -> Dict[str, float]:
    return {doc_id: 1.0/(k+rank) for rank,(doc_id,_) in enumerate(ranked,1)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", required=True)
    ap.add_argument("--questions", default=None)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--out", default="runs/minimal_hybrid_rrf.jsonl")
    ap.add_argument("--bm25_pool", type=int, default=50, help="BM25 상위 n")
    ap.add_argument("--dense_pool", type=int, default=50, help="Dense 상위 n")
    args = ap.parse_args()

    docs = list(iter_corpus(args.corpus_dir))
    if not docs: raise SystemExit(f"No docs under {args.corpus_dir}")
    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]

    # BM25
    bm25 = BM25Okapi([tok(t) for t in texts])

    # Dense + FAISS
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    mat = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    qs = load_questions(args.questions)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if not qs:
        print("질문 파일이 없어요. 인터랙티브 모드(엔터=종료).")
        while True:
            q = input("> ").strip()
            if not q: break
            # 개별 결과
            scores = bm25.get_scores(tok(q))
            btop = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:args.bm25_pool]
            qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            D, I = index.search(qv, min(args.dense_pool, len(ids)))
            dtop = [(ids[i], float(s)) for i,s in zip(I[0].tolist(), D[0].tolist())]
            btop_ids = [(ids[i], float(s)) for i,s in btop]

            # RRF 결합
            bm = rrf(btop_ids); dn = rrf(dtop)
            all_ids = set(bm)|set(dn)
            fused = [(d, bm.get(d,0.0)+dn.get(d,0.0)) for d in all_ids]
            fused.sort(key=lambda x: x[1], reverse=True)
            for rank,(doc_id,score) in enumerate(fused[:args.top_k],1):
                d = docs[ids.index(doc_id)]
                print(f"{rank}. {doc_id}  score={score:.4f}  {d['text'][:200].replace(chr(10),' ')}...")
        return

    with open(args.out, "w", encoding="utf-8") as w:
        for q in tqdm(qs, desc="retrieving"):
            # BM25
            scores = bm25.get_scores(tok(q["question"]))
            btop = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:args.bm25_pool]
            btop_ids = [(ids[i], float(s)) for i,s in btop]
            # Dense
            qv = model.encode([q["question"]], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            D, I = index.search(qv, min(args.dense_pool, len(ids)))
            dtop = [(ids[i], float(s)) for i,s in zip(I[0].tolist(), D[0].tolist())]
            # RRF
            bm = rrf(btop_ids); dn = rrf(dtop)
            all_ids = set(bm)|set(dn)
            fused = [(d, bm.get(d,0.0)+dn.get(d,0.0)) for d in all_ids]
            fused.sort(key=lambda x: x[1], reverse=True)

            hits = []
            for doc_id,score in fused[:args.top_k]:
                d = docs[ids.index(doc_id)]
                hits.append({"doc_id": doc_id, "score": float(score), "snippet": d["text"][:400]})
            rec = {"qid": q["id"], "question": q["question"], "method": "hybrid_rrf",
                   "top_k": args.top_k, "hits": hits}
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"saved -> {args.out}")

if __name__ == "__main__":
    main()
