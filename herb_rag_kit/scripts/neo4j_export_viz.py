from __future__ import annotations
import os, sys
import argparse
from typing import Dict, Any

# __SRC_PATH_HACK__ to import herb_rag_kit without installing the package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from herb_rag_kit.graphdb.neo4j_store import Neo4jStore
from pyvis.network import Network


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cypher", default="MATCH (e1:Entity)-[r:RELATES]->(e2:Entity) RETURN e1, r, e2 LIMIT 200")
    ap.add_argument("--out", default="runs/kg_subgraph.html")
    args = ap.parse_args()

    store = Neo4jStore()
    rows = store.run_cypher(args.cypher)

    net = Network(height="820px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120, spring_strength=0.015)

    node_ids = set()
    def add_node(n: Dict[str, Any], color: str) -> None:
        node_id = n.element_id if hasattr(n, "element_id") else (n.get("id") or n.get("key") or n.get("name"))
        label = n.get("name") or n.get("title") or n.get("id") or "?"
        if node_id not in node_ids:
            net.add_node(node_id, label=label, color=color, title=str(n))
            node_ids.add(node_id)

    for rec in rows:
        e1 = rec.get("e1") or rec.get("e1")
        e2 = rec.get("e2") or rec.get("e2")
        r = rec.get("r")
        if not (e1 and e2 and r):
            # fallback for generic RETURN node, rel, node
            nodes = [v for v in rec.values() if getattr(v, "_properties", None) is not None]
            rels = [v for v in rec.values() if getattr(v, "nodes", None) is not None]
            if len(nodes) >= 2 and rels:
                e1, e2, r = nodes[0], nodes[1], rels[0]
            else:
                continue
        add_node(e1, color="#4e79a7")
        add_node(e2, color="#f28e2b")
        pred = getattr(r, "_properties", {}).get("predicate", "relates") if hasattr(r, "_properties") else (r.get("predicate") if isinstance(r, dict) else "relates")
        conf = getattr(r, "_properties", {}).get("confidence", 1.0) if hasattr(r, "_properties") else (r.get("confidence") if isinstance(r, dict) else 1.0)
        net.add_edge(e1.element_id if hasattr(e1, "element_id") else (e1.get("id") or e1.get("key")),
                     e2.element_id if hasattr(e2, "element_id") else (e2.get("id") or e2.get("key")),
                     label=f"{pred} ({conf:.2f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    try:
        # 일부 환경에서 net.show()가 템플릿 문제로 실패할 수 있어 write_html로 대체
        net.write_html(args.out, notebook=False)
    except Exception:
        # 구버전 호환
        try:
            net.save_graph(args.out)
        except Exception as e:
            raise SystemExit(f"Failed to write html: {e}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()


