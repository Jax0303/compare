#!/usr/bin/env python3
from __future__ import annotations
import os, argparse, json, sys

# __SRC_PATH_HACK__
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC=os.path.join(ROOT,'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from herb_rag_kit.graphdb.neo4j_store import Neo4jStore
from herb_rag_kit.graphdb.schema_stats import compute_schema_stats, save_schema_stats


def main():
    ap = argparse.ArgumentParser(description="Compute soft schema stats from Neo4j graph")
    ap.add_argument("--out", default="runs/fb15k237_schema.json")
    args = ap.parse_args()

    store = Neo4jStore()
    stats = compute_schema_stats(store)
    save_schema_stats(stats, args.out)
    print(json.dumps({"ok": True, "predicates": len(stats)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


