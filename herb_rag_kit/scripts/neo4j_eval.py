import os, json, time, argparse
from typing import List, Dict, Any, Tuple
from herb_rag_kit.graphdb.neo4j_store import Neo4jStore

def precision_recall_at_k(retrieved: List[str], gold: List[str], k: int = 5) -> Tuple[float,float]:
    r = retrieved[:k]; rset, gset = set(r), set(gold)
    tp = len(rset & gset)
    return (tp/max(len(r),1), tp/max(len(gset),1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True, help="JSONL with {'qid','question','gold_doc_ids':[]}")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    store = Neo4jStore()
    print(json.dumps({"graph_size": store.graph_size()}, ensure_ascii=False))

    out = []
    with open(args.questions, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line); qid = d.get("qid") or d.get("id")
            q = d["question"]; gold = d.get("gold_doc_ids", [])

            t0=time.time(); ft = store.fulltext(q, k=args.k); t1=time.time()
            # 간단한 0벡터 예시; 실제로는 질문 임베딩 사용 권장
            qvec = [0.0]*store.cfg.emb_dim
            t2=time.time(); kn = store.knn(qvec, k=args.k); t3=time.time()

            docs_ft = [r["id"] for r in ft]
            docs_kn = [r["id"] for r in kn]
            p_ft, r_ft = precision_recall_at_k(docs_ft, gold, k=args.k)
            p_kn, r_kn = precision_recall_at_k(docs_kn, gold, k=args.k)

            out.append({
                "qid": qid, "q": q, "gold": gold,
                "fulltext_docs": docs_ft, "knn_docs": docs_kn,
                "metrics": {
                    "p@k_fulltext": p_ft, "r@k_fulltext": r_ft,
                    "p@k_knn": p_kn, "r@k_knn": r_kn,
                    "latency_ms": {"fulltext": int((t1-t0)*1000), "knn": int((t3-t2)*1000)}
                }
            })
    print(json.dumps({"eval": out}, ensure_ascii=False))

if __name__ == "__main__":
    main()
