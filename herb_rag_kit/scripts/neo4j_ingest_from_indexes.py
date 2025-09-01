import os, json, argparse
import numpy as np
from herb_rag_kit.graphdb.neo4j_store import Neo4jStore

def iter_docstore(jsonl_path: str):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            did = d.get("id") or d.get("doc_id")
            txt = (d.get("text") or "").strip()
            title = d.get("title") or None
            url = d.get("url") or None
            if did and txt is not None:
                yield did, title, txt, url

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="./indexes/herb",
                    help="HERB 인덱스 디렉토리 (docstore.jsonl + embeddings.npy)")
    ap.add_argument("--limit", type=int, default=0, help="0이면 전체")
    args = ap.parse_args()

    jsonl = os.path.join(args.index_dir, "docstore.jsonl")
    npy = os.path.join(args.index_dir, "embeddings.npy")
    if not os.path.exists(jsonl):
        raise FileNotFoundError(jsonl)
    if not os.path.exists(npy):
        raise FileNotFoundError(npy)

    print(f"[LOAD] {jsonl}")
    print(f"[LOAD] {npy}")
    embs = np.load(npy)  # shape: (N, D)
    store = Neo4jStore()
    store.create_schema()

    n = 0
    for i, (did, title, text, url) in enumerate(iter_docstore(jsonl)):
        if args.limit and n >= args.limit: break
        vec = embs[i].astype("float32").tolist()
        store.upsert_document(did, text, title=title, url=url, embedding=vec)
        n += 1
        if n % 1000 == 0:
            print(f"[INGEST] {n} docs")

    print(f"[DONE] Ingested {n} docs into Neo4j.")

if __name__ == "__main__":
    main()
