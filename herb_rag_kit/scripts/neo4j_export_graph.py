from __future__ import annotations
import os, sys, argparse, csv
from typing import Dict, Any

# __SRC_PATH_HACK__ to import herb_rag_kit without installing the package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from herb_rag_kit.graphdb.neo4j_store import Neo4jStore


def build_graph_from_rows(rows: list[dict[str, Any]]):
    import networkx as nx
    G = nx.DiGraph()

    def node_id(n: Dict[str, Any]) -> str:
        return n.get("id") or n.get("key") or n.get("name") or "?"

    for rec in rows:
        e1 = rec.get("e1") or rec.get("n1")
        e2 = rec.get("e2") or rec.get("n2")
        r = rec.get("r") or rec.get("rel")
        if not (e1 and e2 and r):
            # fallback: pick first two nodes and one rel from generic row
            nodes = [v for v in rec.values() if isinstance(v, dict) and v.get("_properties") is not None]
            rels = [v for v in rec.values() if hasattr(v, "nodes") or (isinstance(v, dict) and v.get("predicate"))]
            if len(nodes) >= 2 and rels:
                e1, e2, r = nodes[0], nodes[1], rels[0]
            else:
                continue

        # py2neo-like dicts
        def props(x):
            if isinstance(x, dict):
                return x.get("_properties") or x
            return getattr(x, "_properties", {})

        p1, p2, pr = props(e1), props(e2), props(r)
        u, v = node_id(p1), node_id(p2)
        if not G.has_node(u):
            G.add_node(u, **{k: p1.get(k) for k in ("name","title","id","key","type") if k in p1})
        if not G.has_node(v):
            G.add_node(v, **{k: p2.get(k) for k in ("name","title","id","key","type") if k in p2})
        G.add_edge(u, v, predicate=pr.get("predicate", "relates"), confidence=pr.get("confidence", 1.0))

    return G


def export_graph(G, out: str, fmt: str):
    import networkx as nx
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fmt = fmt.lower()
    if fmt == "gexf":
        nx.write_gexf(G, out)
    elif fmt == "graphml":
        nx.write_graphml(G, out)
    elif fmt == "csv":
        base, _ = os.path.splitext(out)
        nodes_csv = base + "_nodes.csv"
        edges_csv = base + "_edges.csv"
        with open(nodes_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id","name","title","type","key"]) 
            for n, d in G.nodes(data=True):
                w.writerow([n, d.get("name",""), d.get("title",""), d.get("type",""), d.get("key","")])
        with open(edges_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source","target","predicate","confidence"]) 
            for u, v, d in G.edges(data=True):
                w.writerow([u, v, d.get("predicate","relates"), d.get("confidence",1.0)])
        print(f"Saved: {nodes_csv}, {edges_csv}")
        return
    elif fmt == "png":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=0.4, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=60, node_color="#4e79a7", alpha=0.8)
        nx.draw_networkx_edges(G, pos, arrows=False, width=0.5, alpha=0.4)
        # Labels for small graphs only
        if G.number_of_nodes() <= 200:
            labels = {n: (d.get("name") or d.get("title") or str(n)) for n,d in G.nodes(data=True)}
            nx.draw_networkx_labels(G, pos, labels, font_size=6)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()
    else:
        raise SystemExit(f"Unsupported format: {fmt}")
    print(f"Saved: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cypher", default="MATCH (e1:Entity)-[r:RELATES]->(e2:Entity) RETURN e1, r, e2 LIMIT 200")
    ap.add_argument("--out", required=True, help="Output path (extension ignored for CSV pair)")
    ap.add_argument("--format", choices=["gexf","graphml","csv","png"], default="gexf")
    args = ap.parse_args()

    store = Neo4jStore()
    rows = store.run_cypher(args.cypher)
    G = build_graph_from_rows(rows)
    export_graph(G, args.out, args.format)


if __name__ == "__main__":
    main()



