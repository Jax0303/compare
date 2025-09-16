import os, json, glob, time
from herb_rag_kit.graphdb.neo4j_store import Neo4jStore
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

def main():
    store = Neo4jStore()
    store.create_schema()
    model = None
    model_name = os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")
    if SentenceTransformer is not None:
        try:
            model = SentenceTransformer(model_name)
            model.encode(["warmup"])  # preload
            print({"model":"ready","name":model_name})
        except Exception as e:
            print({"model_err":str(e)})
            model = None

    paths = sorted(glob.glob("indexes/txt/*.jsonl"))
    print({"docstores": paths}); total=0; t0=time.time()
    batch, ids, texts, titles = [], [], [], []
    def flush_batch():
        nonlocal total, batch, ids, texts, titles
        if not batch:
            return
        embs = [None]*len(batch)
        if model is not None:
            try:
                embs_np = model.encode(batch, normalize_embeddings=True)
                for i,v in enumerate(embs_np):
                    try:
                        embs[i] = v.astype("float32").tolist()
                    except Exception:
                        embs[i] = [float(x) for x in v]
            except Exception:
                pass
        for i in range(len(batch)):
            try:
                store.upsert_document(doc_id=ids[i], text=texts[i], title=titles[i], embedding=embs[i])
            except Exception as e:
                print({"upsert_err": ids[i], "err": str(e)})
            total += 1
        print({"progress": total})
        batch, ids, texts, titles = [], [], [], []

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for ln in f:
                try:
                    d=json.loads(ln)
                except Exception:
                    continue
                if "id" not in d or "text" not in d:
                    continue
                ids.append(d["id"]); texts.append(d["text"]); titles.append(d.get("title"))
                batch.append(d["text"][:2000])
                if len(batch) >= int(os.getenv("UPSERT_BATCH", "32")):
                    flush_batch()
    flush_batch()
    print({"upserted_total": total, "elapsed_sec": round(time.time()-t0,2)})

if __name__ == "__main__":
    main()

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
