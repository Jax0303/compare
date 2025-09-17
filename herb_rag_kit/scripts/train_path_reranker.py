#!/usr/bin/env python3
from __future__ import annotations
import os, json, argparse, random, sys

# __SRC_PATH_HACK__
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC=os.path.join(ROOT,'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from typing import List, Dict, Any
from herb_rag_kit.graphdb.schema_stats import load_schema_stats
from herb_rag_kit.retrieval.path_reranker import _feature_vector


def iter_session(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            try:
                yield json.loads(ln)
            except Exception:
                continue


def build_training_data(session_path: str, schema_path: str):
    schema = load_schema_stats(schema_path)
    X: List[List[float]] = []
    y: List[float] = []
    for row in iter_session(session_path):
        rer = row.get("reranked_rows", [])
        paths = row.get("paths", [])
        # 약지도: 상위 k/2는 1, 나머지는 0 (없으면 graph_rows를 사용)
        cand = rer or (row.get("graph_rows", []) + row.get("hybrid_rows", []))
        if not cand:
            continue
        half = max(1, len(cand)//2)
        for i, it in enumerate(cand):
            X.append(_feature_vector(it, schema, paths))
            y.append(1.0 if i < half else 0.0)
    return X, y


def main():
    ap = argparse.ArgumentParser(description="Train LightGBM reranker from session logs")
    ap.add_argument("--session", default="runs/session.jsonl")
    ap.add_argument("--schema", default="runs/fb15k237_schema.json")
    ap.add_argument("--out", default="runs/path_reranker.lgb")
    args = ap.parse_args()

    X, y = build_training_data(args.session, args.schema)
    if not X:
        raise SystemExit("no training data from session logs")
    try:
        import lightgbm as lgb
        import numpy as np
        dtrain = lgb.Dataset(np.array(X, dtype="float32"), label=np.array(y, dtype="float32"))
        params = {
            "objective": "binary",
            "metric": ["auc"],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
        }
        bst = lgb.train(params, dtrain, num_boost_round=200)
        bst.save_model(args.out)
        print(json.dumps({"ok": True, "n": len(y), "model": args.out}, ensure_ascii=False))
    except Exception as e:
        raise SystemExit(f"lightgbm training failed: {e}")


if __name__ == "__main__":
    main()


