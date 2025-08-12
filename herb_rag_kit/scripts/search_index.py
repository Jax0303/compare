#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json, argparse, pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣_]+")
def tok(s: str) -> List[str]:
    return TOKEN_RE.findall((s or "").lower())

def rrf(ranked: List[Tuple[str, float]], k: int = 60) -> dict:
    return {doc: 1.0/(k+rank) for rank,(doc,_) in enumerate(ranked,1)}

def load_docstore(p: Path) -> List[Dict]:
    docs=[]
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="indexes/herb")
    ap.add_argument("--questions", default=None, help="JSONL {'id','question'}")
    ap.add_argument("--method", choices=["bm25","dense","hybrid"], default="hybrid")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--out", default="runs/search.jsonl")
    args = ap.parse_args()

    idx = Path(args.index_dir)
    ids = json.load(open(idx/"ids.json","r",encoding="utf-8"))
    docs = load_docstore(idx/"docstore.jsonl")

    # BM25 로드
    with open(idx/"bm25.pkl","rb") as f: bm25 = pickle.load(f)

    # Dense 로드
    import faiss
    faiss_idx = faiss.read_index(str(idx/"faiss.index"))
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def search_one(q: str) -> List[Tuple[str,float]]:
        out = []
        if args.method in ("bm25","hybrid"):
            scores = bm25.get_scores(tok(q))
            btop = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:max(args.top_k,50)]
            btop = [(ids[i], float(s)) for i,s in btop]
        else:
            btop = []

        if args.method in ("dense","hybrid"):
            qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            D, I = faiss_idx.search(qv, max(args.top_k,50))
            dtop = [(ids[i], float(s)) for i,s in zip(I[0].tolist(), D[0].tolist())]
        else:
            dtop = []

        if args.method == "bm25": return btop[:args.top_k]
        if args.method == "dense": return dtop[:args.top_k]
        # hybrid RRF
        bm = rrf(btop); dn = rrf(dtop)
        keys = set(bm)|set(dn)
        fused = [(k, bm.get(k,0.0)+dn.get(k,0.0)) for k in keys]
        fused.sort(key=lambda x:x[1], reverse=True)
        return fused[:args.top_k]

    if not args.questions:
        print("인터랙티브(엔터=종료):")
        while True:
            q = input("> ").strip()
            if not q: break
            hits = search_one(q)
            for rank,(doc_id,score) in enumerate(hits,1):
                d = docs[ids.index(doc_id)]
                print(f"{rank}. {doc_id}  {score:.4f}  {d['text'][:200].replace(chr(10),' ')}...")
        return

    # 배치
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.questions, "r", encoding="utf-8") as f, open(args.out,"w",encoding="utf-8") as w:
        for line in f:
            o = json.loads(line)
            qid = o.get("id") or o.get("_id") or o.get("qid")
            qtx = o.get("question") or o.get("query") or o.get("q")
            if not qtx: continue
            hits = search_one(qtx)
            out_hits=[]
            for doc_id,score in hits:
                d = docs[ids.index(doc_id)]
                out_hits.append({"doc_id": doc_id, "score": float(score), "snippet": d["text"][:400]})
            rec = {"qid": qid, "question": qtx, "method": args.method, "top_k": args.top_k, "hits": out_hits}
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"saved -> {args.out}")

if __name__ == "__main__":
    main()
