from __future__ import annotations
import os, sys, json
from typing import Dict, Any, List

# __SRC_PATH_HACK__ to import herb_rag_kit without installing the package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from herb_rag_kit.graphdb.neo4j_store import Neo4jStore


def gini(values: List[int]) -> float:
    vals = sorted([max(0, int(v)) for v in values])
    n = len(vals)
    if n == 0:
        return 0.0
    s = sum(vals)
    if s == 0:
        return 0.0
    cum = 0
    for i, v in enumerate(vals, 1):
        cum += i * v
    return (2 * cum) / (n * s) - (n + 1) / n


def main() -> None:
    store = Neo4jStore()

    # Basic sizes
    sizes = store.graph_size()

    # Confidence stats on RELATES
    cy_conf = (
        """
        MATCH ()-[r:RELATES]->()
        RETURN count(r) AS n,
               sum(CASE WHEN coalesce(r.confidence,1.0) < 0.6 THEN 1 ELSE 0 END) AS low,
               avg(coalesce(r.confidence,1.0)) AS avg_conf
        """
    )
    conf_row = store.run_cypher(cy_conf)[0] if sizes.get("relations", 0) else {"n": 0, "low": 0, "avg_conf": 0.0}

    # Outdegree distribution (hubness)
    cy_outdeg = (
        """
        MATCH (s:Entity)-[:RELATES]->()
        RETURN s.key AS key, count(*) AS deg
        """
    )
    outdeg_rows = store.run_cypher(cy_outdeg)
    out_degs = [int(r["deg"]) for r in outdeg_rows]

    # Predicate distribution (skew/bias)
    cy_pred = (
        """
        MATCH ()-[r:RELATES]->()
        RETURN r.predicate AS p, count(*) AS c
        """
    )
    pred_rows = store.run_cypher(cy_pred)
    pred_counts = [int(r["c"]) for r in pred_rows]
    pred_top = sorted(pred_counts, reverse=True)[:1]
    pred_total = sum(pred_counts) or 1
    top_pred_share = (pred_top[0] / pred_total) if pred_top else 0.0

    # Objects per (s,p)
    cy_sp_obj = (
        """
        MATCH (s:Entity)-[r:RELATES]->(o:Entity)
        WITH s.key AS sk, r.predicate AS p, count(DISTINCT o.key) AS num_o
        RETURN num_o AS n
        """
    )
    sp_rows = store.run_cypher(cy_sp_obj)
    sp_counts = [int(r["n"]) for r in sp_rows]
    sp_gini = gini(sp_counts)

    out = {
        "sizes": sizes,
        "confidence": {
            "total_relations": int(conf_row.get("n", 0)),
            "low_conf_relations": int(conf_row.get("low", 0)),
            "low_conf_rate": (int(conf_row.get("low", 0)) / max(1, int(conf_row.get("n", 0)))),
            "avg_confidence": float(conf_row.get("avg_conf", 0.0)),
        },
        "hubness": {
            "outdegree_gini": gini(out_degs),
            "p95_outdegree": int(sorted(out_degs)[int(0.95 * len(out_degs))]) if out_degs else 0,
            "max_outdegree": max(out_degs) if out_degs else 0,
        },
        "predicate_bias": {
            "num_predicates": len(pred_counts),
            "gini": gini(pred_counts),
            "top_pred_share": top_pred_share,
        },
        "sp_object_concentration": {
            "gini": sp_gini,
        },
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


