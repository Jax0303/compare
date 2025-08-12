#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, glob, json, argparse
from typing import List, Dict
from tqdm import tqdm
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣_]+")
def tokenize(s: str) -> List[str]:
    return TOKEN_RE.findall((s or "").lower())

def iter_corpus(corpus_dir: str):
    paths = glob.glob(os.path.join(corpus_dir, "*.json"))
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        text = (d.get("text") or d.get("content") or d.get("message") or "").strip()
        if not text:
            continue
        doc_id = d.get("id") or os.path.basename(p)
        yield {"id": str(doc_id), "text": text}

def load_questions(path_jsonl: str) -> List[Dict]:
    if not path_jsonl:
        return []
    qs = []
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            qid = o.get("id") or o.get("_id") or o.get("qid")
            qtx = o.get("question") or o.get("query") or o.get("q")
            if qtx:
                qs.append({"id": qid, "question": qtx})
    return qs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", required=True, help="e.g., ~/HERB/data/corpus")
    ap.add_argument("--questions", default=None, help="JSONL: {'id','question'} per line")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--out", default="runs/minimal_bm25.jsonl")
    args = ap.parse_args()

    docs = list(iter_corpus(args.corpus_dir))
    if not docs:
        raise SystemExit(f"No docs under {args.corpus_dir}")

    tokens = [tokenize(d["text"]) for d in docs]
    bm25 = BM25Okapi(tokens)

    qs = load_questions(args.questions)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if not qs:
        print("질문 파일이 없네요. 인터랙티브 모드로 전환합니다. (엔터만 누르면 종료)")
        while True:
            q = input("> ").strip()
            if not q:
                break
            scores = bm25.get_scores(tokenize(q))
            top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:args.top_k]
            for rank, (idx, score) in enumerate(top, 1):
                d = docs[idx]
                snippet = d["text"][:200].replace("\n", " ")
                print(f"{rank}. {d['id']}  score={score:.3f}  {snippet}...")
        return

    with open(args.out, "w", encoding="utf-8") as w:
        for q in tqdm(qs, desc="retrieving"):
            scores = bm25.get_scores(tokenize(q["question"]))
            top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:args.top_k]
            hits = []
            for i, s in top:
                d = docs[i]
                hits.append({
                    "doc_id": d["id"],
                    "score": float(s),
                    "snippet": d["text"][:400]
                })
            rec = {
                "qid": q["id"],
                "question": q["question"],
                "method": "bm25",
                "top_k": args.top_k,
                "hits": hits
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"saved -> {args.out}")

if __name__ == "__main__":
    main()
