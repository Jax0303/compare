#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, glob, json, argparse
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", required=True)
    ap.add_argument("--questions", default=None)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--out", default="runs/minimal_dense.jsonl")
    args = ap.parse_args()

    docs = list(iter_corpus(args.corpus_dir))
    if not docs: raise SystemExit(f"No docs under {args.corpus_dir}")
    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]

    from sentence_transformers import SentenceTransformer
    import faiss

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
            qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            D, I = index.search(qv, args.top_k)
            for rank, (i, s) in enumerate(zip(I[0], D[0]), 1):
                d = docs[int(i)]
                print(f"{rank}. {d['id']}  score={float(s):.3f}  {d['text'][:200].replace(chr(10),' ')}...")
        return

    with open(args.out, "w", encoding="utf-8") as w:
        for q in tqdm(qs, desc="retrieving"):
            qv = model.encode([q["question"]], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            D, I = index.search(qv, min(args.top_k, len(ids)))
            hits = []
            for i, s in zip(I[0].tolist(), D[0].tolist()):
                d = docs[int(i)]
                hits.append({"doc_id": d["id"], "score": float(s), "snippet": d["text"][:400]})
            rec = {"qid": q["id"], "question": q["question"], "method": "dense", "top_k": args.top_k, "hits": hits}
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"saved -> {args.out}")

if __name__ == "__main__":
    main()
