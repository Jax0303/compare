from __future__ import annotations
import os, json, time, argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
from ..store.document_store import Document, DocumentStore
from ..store.bm25_index import BM25Index
from ..store.embedding_index import EmbeddingIndex
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.rerank import simple_rerank
from ..llm.gemini_client import embed_texts, embed_query, generate
from ..methods.dragin import rind_should_retrieve, qfs_build_query
from ..methods.dota import route_bucket, hybrid_stage
from ..methods.graphrag import GraphRAG
from ..methods.hdrag import build_hybrid_representation

def load_corpus(herb_root: str) -> Tuple[DocumentStore, list[str], list[str]]:
    store = DocumentStore()
    corpus_dir = os.path.join(herb_root, "data", "corpus")
    doc_ids, texts = [], []
    for fn in os.listdir(corpus_dir):
        if not fn.endswith(".json"): 
            continue
        p = os.path.join(corpus_dir, fn)
        try:
            doc = json.load(open(p, "r", encoding="utf-8"))
            d = Document(doc_id=str(doc.get("doc_id") or fn[:-5]), text=doc.get("text",""), timestamp=doc.get("timestamp"), metadata=doc.get("metadata",{}))
            store.add(d)
            doc_ids.append(d.doc_id)
            texts.append(d.text)
        except Exception:
            continue
    return store, doc_ids, texts

def build_indexes(doc_ids: list[str], texts: list[str]):
    bm25 = BM25Index(doc_ids, texts)
    # 임베딩 진행률 표시 및 shape 보정
    doc_embs = np.asarray(list(tqdm(embed_texts(texts), total=len(texts), desc='Embedding')))
    print(f"[DEBUG] doc_embs.shape: {doc_embs.shape}")
    print(f"[DEBUG] doc_ids: {len(doc_ids)}, texts: {len(texts)}")
    if doc_embs.ndim == 1:
        doc_embs = doc_embs.reshape(1, -1)
    elif doc_embs.ndim > 2:
        doc_embs = doc_embs.squeeze()
    print(f"[DEBUG] doc_embs.shape after squeeze/reshape: {doc_embs.shape}")
    emb_index = EmbeddingIndex(doc_ids, doc_embs)
    return bm25, emb_index

def run_dragin(question: str, bm25, emb_index, store: DocumentStore, k: int = 5) -> Dict[str, Any]:
    start = time.time()
    retrieved = []
    if rind_should_retrieve(question):
        q = qfs_build_query(question)
        qvec = np.array(embed_query(q))
        # simple hybrid then rerank with content
        hits = []
        for d,s in HybridRetriever(bm25, emb_index, alpha=0.5).search(q, qvec, k=k*2):
            txt = store.get(d).text if store.get(d) else ""
            hits.append((d,s,txt))
        reranked = simple_rerank(question, hits, top_k=k)
        retrieved = [{"doc_id": d, "score": float(s), "timestamp": (store.get(d).timestamp if store.get(d) else None)} for d,s,_ in reranked]
    context = "\n\n".join([store.get(r["doc_id"]).text for r in retrieved if store.get(r["doc_id"])][:3])
    prompt = f"Question: {question}\nContext (may be empty):\n{context}\nAnswer briefly and factually:"
    ans = generate(prompt, temperature=0.2, max_output_tokens=256)
    return {"pred": ans, "retrieved": retrieved, "latency_ms": (time.time()-start)*1000}

def run_dota(question: str, bm25, emb_index, store: DocumentStore, k: int = 5) -> Dict[str, Any]:
    start = time.time()
    bucket = route_bucket(question, buckets=["code","slack","docs","pr","people"])
    # for demo, route only changes alpha weight lightly
    alpha = {"code":0.3, "slack":0.4, "docs":0.5, "pr":0.6, "people":0.5}.get(bucket, 0.5)
    qvec = np.array(embed_query(question))
    ranked = HybridRetriever(bm25, emb_index, alpha=alpha).search(question, qvec, k=k*2)
    hits = [(d,s, store.get(d).text if store.get(d) else "") for d,s in ranked]
    reranked = simple_rerank(question, hits, top_k=k)
    retrieved = [{"doc_id": d, "score": float(s), "timestamp": (store.get(d).timestamp if store.get(d) else None)} for d,s,_ in reranked]
    context = "\n\n".join([store.get(r["doc_id"]).text for r in retrieved if store.get(r["doc_id"])][:3])
    ans = generate(f"Q: {question}\nContext:\n{context}\nA:", temperature=0.2, max_output_tokens=256)
    return {"pred": ans, "retrieved": retrieved, "latency_ms": (time.time()-start)*1000, "bucket": bucket}

def run_graphrag(question: str, graph: GraphRAG, store: DocumentStore, k: int = 5) -> Dict[str, Any]:
    start = time.time()
    doc_ids = graph.retrieve(question, hops=2, k=k)
    retrieved = [{"doc_id": d, "score": 1.0, "timestamp": (store.get(d).timestamp if store.get(d) else None)} for d in doc_ids]
    context = "\n\n".join([store.get(d).text for d in doc_ids if store.get(d)][:3])
    ans = generate(f"Q: {question}\nGraph context:\n{context}\nA:", temperature=0.2, max_output_tokens=256)
    return {"pred": ans, "retrieved": retrieved, "latency_ms": (time.time()-start)*1000}

def run_hdrag(question: str, bm25, emb_index, store: DocumentStore, k: int = 5) -> Dict[str, Any]:
    start = time.time()
    # Stage1: ensemble
    qvec = np.array(embed_query(question))
    ranked = HybridRetriever(bm25, emb_index, alpha=0.5).search(question, qvec, k=k*3)
    # Stage2: LLM-based re-score
    rescored = []
    for d, s in ranked:
        txt = store.get(d).text if store.get(d) else ""
        score_txt = generate(f"Question: {question}\nDoc snippet: {txt[:800]}\nOn 0-1 scale, how relevant is this doc for answering? Return only a number.", temperature=0.0, max_output_tokens=8).strip()
        try:
            s2 = float(score_txt)
        except:
            s2 = 0.0
        rescored.append((d, 0.5*s + 0.5*s2, txt))
    reranked = sorted(rescored, key=lambda x:x[1], reverse=True)[:k]
    retrieved = [{"doc_id": d, "score": float(s), "timestamp": (store.get(d).timestamp if store.get(d) else None)} for d,s,_ in reranked]
    context = "\n\n".join([store.get(r["doc_id"]).text for r in retrieved if store.get(r["doc_id"])][:3])
    ans = generate(f"Q: {question}\nHybrid doc context:\n{context}\nA:", temperature=0.2, max_output_tokens=256)
    return {"pred": ans, "retrieved": retrieved, "latency_ms": (time.time()-start)*1000}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--herb_root", required=True)
    ap.add_argument("--index", required=False, help="(reserved)")
    ap.add_argument("--method", required=True, choices=["dragin","dota","graphrag","hdrag"])
    ap.add_argument("--questions", default=None, help="path to questions.jsonl (default: HERB/data/questions.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    import time
    print("[LOG] Loading corpus...")
    t0 = time.time()
    store, doc_ids, texts = load_corpus(args.herb_root)
    print(f"[LOG] Corpus loaded in {time.time()-t0:.2f}s, {len(doc_ids)} docs")

    print("[LOG] Building indexes...")
    t1 = time.time()
    bm25, emb_index = build_indexes(doc_ids, texts)
    print(f"[LOG] Indexes built in {time.time()-t1:.2f}s")

    print("[LOG] Building GraphRAG...")
    t2 = time.time()
    graph = GraphRAG()
    for d in store.all():
        graph.add_document(d.doc_id, d.text)
    print(f"[LOG] GraphRAG built in {time.time()-t2:.2f}s")

    qpath = args.questions or os.path.join(args.herb_root, "data", "questions.jsonl")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[LOG] Start processing questions from {qpath}")
    tq = time.time()
    with open(args.out, "w", encoding="utf-8") as fout:
        for i, line in enumerate(open(qpath, "r", encoding="utf-8")):
            if not line.strip():
                continue
            q = json.loads(line)
            qid = str(q.get("qid") or q.get("id"))
            question = q.get("question") or q.get("query") or ""
            if not question:
                continue
            print(f"[LOG] Q{i+1} (qid={qid}): {question[:50]}...")
            tq1 = time.time()
            if args.method == "dragin":
                res = run_dragin(question, bm25, emb_index, store, k=args.k)
            elif args.method == "dota":
                res = run_dota(question, bm25, emb_index, store, k=args.k)
            elif args.method == "graphrag":
                res = run_graphrag(question, graph, store, k=args.k)
            else:
                res = run_hdrag(question, bm25, emb_index, store, k=args.k)
            print(f"[LOG] Q{i+1} 처리 시간: {time.time()-tq1:.2f}s, latency_ms={res.get('latency_ms',-1):.1f}")
            out = {"qid": qid, **res}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"[LOG] 전체 질문 처리 완료: {time.time()-tq:.2f}s")

if __name__ == "__main__":
    main()
