import os, json, glob, time
from herb_rag_kit.graphdb.neo4j_store import Neo4jStore
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

store = Neo4jStore()
store.create_schema()

model = None
if SentenceTransformer is not None:
    try:
        model = SentenceTransformer(os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2"))
        model.encode(["warmup"])
        print({"model":"ready"})
    except Exception as e:
        print({"model_err": str(e)})

paths = sorted(glob.glob("indexes/txt/*.jsonl"))
print({"docstores": paths}); total=0; t0=time.time()

for p in paths:
    with open(p, encoding="utf-8") as f:
        for ln in f:
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if "id" not in d or "text" not in d:
                continue
            emb = None
            if model is not None:
                try:
                    v = model.encode([d["text"][:2000]], normalize_embeddings=True)[0]
                    try: emb = v.astype("float32").tolist()
                    except Exception: emb = [float(x) for x in v]
                except Exception: pass
            try:
                store.upsert_document(doc_id=d["id"], text=d["text"], title=d.get("title"), embedding=emb)
            except Exception as e:
                print({"upsert_err": d.get("id"), "err": str(e)})
            total += 1
            if total % 100 == 0: print({"progress": total})
print({"upserted_total": total, "elapsed_sec": round(time.time()-t0,2)})
