import argparse, json, time, os, math
from typing import Dict, Any, List, Optional
from herb_rag_kit.graphdb.neo4j_store import Neo4jStore

# --- Query embedding helpers -------------------------------------------------

def try_gemini_embed(q: str, dim: int) -> Optional[List[float]]:
    """Use your herb_rag_kit.llm.gemini_client.embed_query if available.
       Return None when unavailable or dimension mismatch."""
    try:
        from herb_rag_kit.llm.gemini_client import embed_query
        v = embed_query(q)
        if v is None:
            return None
        # to list of floats
        v = v.tolist() if hasattr(v, "tolist") else v
        v = [float(x) for x in v]
        # dim check
        if len(v) != dim:
            return None
        # finite & non-zero norm check
        if not v or not all(math.isfinite(x) for x in v):
            return None
        if sum(x*x for x in v) <= 0.0:
            return None
        return v
    except Exception:
        return None

def try_st_embed(q: str, dim: int) -> Optional[List[float]]:
    """Fallback: sentence-transformers (defaults to MiniLM-L6-v2; 384-dim)."""
    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)
        v = model.encode([q], normalize_embeddings=True)[0]  # unit-norm
        if v.shape[0] != dim:
            return None
        v = v.astype("float32").tolist()
        return v
    except Exception:
        return None

def get_query_vec(question: str, dim: int) -> Optional[List[float]]:
    v = try_gemini_embed(question, dim)
    if v is not None:
        return v
    v = try_st_embed(question, dim)
    if v is not None:
        return v
    return None

# --- Text2Cypher -------------------------------------------------------------

TEXT2CYPHER_SYSTEM = (
    "You are a Cypher assistant. Only output a single read-only Cypher query using MATCH/RETURN. "
    "Do NOT use CREATE/MERGE/SET/DELETE. "
    "Schema: Nodes => (Document {id,title,text,url,embedding}), (Entity {name,type,embedding}); "
    "Rels => (Entity)-[:RELATES {predicate,confidence}]->(Entity), "
    "(Entity)-[:APPEARS_IN]->(Document). "
    "Avoid using APPEARS_IN unless the user explicitly asks about documents; prefer RELATES patterns."
)

def text2cypher(question: str) -> str:
    try:
        from herb_rag_kit.llm.gemini_client import generate
        prompt = f"{TEXT2CYPHER_SYSTEM}\n\nQ: {question}\nCypher:"
        out = generate(prompt, temperature=0.0, max_output_tokens=256).strip()
        out = out.strip("`").strip()
        if out.lower().startswith("cypher"):
            out = out.split("\n", 1)[-1]
        if not out.lower().startswith("match"):
            raise ValueError("not read-only")
        return out
    except Exception:
        # safe fallback
        return "MATCH (d:Document) RETURN d.id AS id, d.title AS title LIMIT 5"

# --- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--mode", choices=["graph","hybrid"], default="graph")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    store = Neo4jStore()
    dim = store.cfg.emb_dim

    t0 = time.time()
    cy = text2cypher(args.question)
    t1 = time.time()
    graph_rows = store.run_cypher(cy)
    t2 = time.time()

    hybrid_rows = []
    if args.mode == "hybrid":
        qvec = get_query_vec(args.question, dim)
        if qvec is None:
            print("[WARN] No valid query embedding (dim mismatch or model unavailable). Skipping kNN.")
        else:
            hybrid_rows = store.knn(qvec, k=args.k)
    t3 = time.time()

    print(json.dumps({
        "question": args.question,
        "cypher": cy,
        "graph_rows": graph_rows[:args.k],
        "hybrid_rows": hybrid_rows,
        "timings_ms": {
            "text2cypher": int((t1 - t0) * 1000),
            "cypher": int((t2 - t1) * 1000),
            "knn": int((t3 - t2) * 1000) if args.mode == "hybrid" else 0,
            "end2end": int((t3 - t0) * 1000) if args.mode == "hybrid" else int((t2 - t0) * 1000),
        }
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
