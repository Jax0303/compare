from __future__ import annotations
import os, json
from typing import Dict, Any
from .neo4j_store import Neo4jStore


def compute_schema_stats(store: Neo4jStore) -> Dict[str, Any]:
    """
    Neo4j에 적재된 FB15K‑237 그래프에서 predicate별 소프트 스키마 통계를 계산.
    - support: triple 수
    - avg_obj_per_sp: (s,p)당 고유 o 평균(기능성 지표)
    - avg_subj_per_po: (p,o)당 고유 s 평균(역-기능성 지표)
    - reciprocity: 양방향 동일 predicate 출현 비율(대칭성 근사)
    """
    stats: Dict[str, Any] = {}
    # support
    rows = store.run_cypher(
        """
        MATCH ()-[r:RELATES]->()
        RETURN r.predicate AS p, count(r) AS c
        """
    )
    for r in rows:
        p = r.get("p") or ""
        if not p: continue
        stats[p] = {"support": int(r.get("c", 0))}

    # avg_obj_per_sp
    rows = store.run_cypher(
        """
        MATCH (s:Entity)-[r:RELATES]->(o:Entity)
        WITH s.key AS sk, r.predicate AS p, collect(DISTINCT o.key) AS oks
        RETURN p, avg(size(oks)) AS avg_o
        """
    )
    for r in rows:
        p = r.get("p")
        if p in stats:
            stats[p]["avg_obj_per_sp"] = float(r.get("avg_o", 0.0))

    # avg_subj_per_po
    rows = store.run_cypher(
        """
        MATCH (s:Entity)-[r:RELATES]->(o:Entity)
        WITH o.key AS ok, r.predicate AS p, collect(DISTINCT s.key) AS sks
        RETURN p, avg(size(sks)) AS avg_s
        """
    )
    for r in rows:
        p = r.get("p")
        if p in stats:
            stats[p]["avg_subj_per_po"] = float(r.get("avg_s", 0.0))

    # reciprocity
    rows = store.run_cypher(
        """
        MATCH (a:Entity)-[r:RELATES]->(b:Entity)
        MATCH (b)-[r2:RELATES {predicate:r.predicate}]->(a)
        RETURN r.predicate AS p, count(*) AS two_way
        """
    )
    for r in rows:
        p = r.get("p")
        two_way = float(r.get("two_way", 0.0))
        support = float(stats.get(p, {}).get("support", 1.0))
        if p in stats:
            stats[p]["reciprocity"] = (two_way / max(1.0, support))

    return stats


def save_schema_stats(stats: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def load_schema_stats(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


