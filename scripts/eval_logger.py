#!/usr/bin/env python3
from __future__ import annotations
import os, json, time, argparse, uuid

def append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as wf:
        wf.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main() -> None:
    ap = argparse.ArgumentParser(description="Simple experiment/evaluation logger (JSONL append)")
    ap.add_argument("--name", required=True, help="experiment name")
    ap.add_argument("--metrics", required=True, help='JSON string, e.g., {"hit@5":0.42}')
    ap.add_argument("--params", default="{}", help='JSON string of parameters')
    ap.add_argument("--out", default="runs/experiments.jsonl")
    args = ap.parse_args()

    rid = str(uuid.uuid4())[:8]
    now = int(time.time())
    try:
        metrics = json.loads(args.metrics)
    except Exception:
        metrics = {"raw": args.metrics}
    try:
        params = json.loads(args.params)
    except Exception:
        params = {"raw": args.params}

    rec = {
        "id": rid,
        "ts": now,
        "name": args.name,
        "metrics": metrics,
        "params": params,
        "env": {
            "GEMINI_MODEL": os.getenv("GEMINI_MODEL"),
            "RERANK_GRAPH_WEIGHT": os.getenv("RERANK_GRAPH_WEIGHT"),
            "RERANK_KNN_WEIGHT": os.getenv("RERANK_KNN_WEIGHT"),
        },
    }
    append_jsonl(args.out, rec)
    print(json.dumps(rec, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()


