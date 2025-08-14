from __future__ import annotations
import os, json, time, argparse, logging
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from ..store.document_store import Document, DocumentStore
from ..store.bm25_index import BM25Index
from ..store.embedding_index import EmbeddingIndex
from ..methods.hdrag import build_hybrid_representation
from ..methods.graphrag import GraphRAG
from ..llm.gemini_client import embed_texts, embed_query, generate

def _coerce_vec(v, dim: int) -> "np.ndarray":
    import numpy as np
    # dict 응답이면 대표 키에서 꺼내기
    if isinstance(v, dict):
        for k in ("embedding","values","vector","data"):
            if k in v:
                v = v[k]; break
    # 리스트/튜플/ndarray 모두 1D로 평탄화
    v = np.asarray(v, dtype="float32")
    if v.ndim > 1:
        v = v.reshape(-1)
    # 비어있거나 유한값 아님 → 0벡터
    if v.size == 0:
        return np.zeros(dim, dtype="float32")
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    # 길이 맞추기(>dim 자르기, <dim 0-padding)
    if v.size != dim:
        if v.size > dim:
            v = v[:dim]
        else:
            v = np.pad(v, (0, dim - v.size))
    return v

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
LOG = logging.getLogger("pipeline")

def load_corpus(herb_root: str) -> Tuple[DocumentStore, List[str], List[str]]:
    store = DocumentStore()
    corpus_dir = os.path.join(herb_root, "data", "corpus")
    if not os.path.isdir(corpus_dir):
        raise SystemExit(f"Corpus dir not found: {corpus_dir}")
    LOG.info("[LOG] Loading corpus...")
    t0 = time.time()
    doc_ids, texts = [], []
    for fn in os.listdir(corpus_dir):
        if not fn.endswith(".json"):
            continue
        fp = os.path.join(corpus_dir, fn)
        try:
            d = json.load(open(fp, "r", encoding="utf-8"))
            did = str(d.get("doc_id") or os.path.splitext(fn)[0])
            txt = d.get("text") or ""
            ts  = d.get("timestamp")
            store.add(Document(doc_id=did, text=txt, timestamp=ts, metadata=d.get("metadata") or {}))
            doc_ids.append(did); texts.append(txt)
        except Exception:
            continue
    LOG.info("[LOG] Corpus loaded in %.2fs, %d docs", time.time()-t0, len(doc_ids))
    return store, doc_ids, texts

def build_indexes(doc_ids: List[str], texts: List[str]) -> Tuple[BM25Index, EmbeddingIndex]:
    if not doc_ids:
        raise SystemExit(
            "No documents found under <HERB_ROOT>/data/corpus.\n"
            "→ Run: python scripts/ingest_herb_hf.py --out_dir <HERB_ROOT>/data"
        )
    LOG.info("[LOG] Building indexes...")
    t0 = time.time()
    cache = Path(".cache"); cache.mkdir(exist_ok=True)
    ids_p, vec_p = cache/"doc_ids.json", cache/"embeddings.npy"

    cached_ids: List[str] = []
    cached_vecs: Optional[np.ndarray] = None
    if ids_p.exists() and vec_p.exists():
        try:
            cached_ids = json.loads(ids_p.read_text())
            cached_vecs = np.load(vec_p)
        except Exception:
            cached_ids, cached_vecs = [], None

    # 완전 일치 시 바로 반환
    if cached_ids == doc_ids and cached_vecs is not None:
        bm25 = BM25Index(doc_ids, texts)
        emb_index = EmbeddingIndex(doc_ids, cached_vecs)
        LOG.info("[LOG] Indexes loaded from cache")
        return bm25, emb_index

    # 부분 캐시 활용: 기존 임베딩을 재사용하고 새 문서만 계산
    id_to_emb = {}
    if cached_vecs is not None and len(cached_ids) == len(cached_vecs):
        id_to_emb = {did: emb for did, emb in zip(cached_ids, cached_vecs)}

    final_embs: List[Optional[np.ndarray]] = []
    new_docs: List[Tuple[int, str]] = []
    for idx, (did, txt) in enumerate(zip(doc_ids, texts)):
        emb = id_to_emb.get(did)
        if emb is not None:
            final_embs.append(emb)
        else:
            final_embs.append(None)
            new_docs.append((idx, (txt or "")[:4000]))

    if new_docs:
        # 1) 입력 방어: 빈 텍스트는 대체
        payload = [(t if t and t.strip() else "[EMPTY]") for _, t in new_docs]

        # 2) 임베딩 호출
        raw = embed_texts(payload)

        # 3) 차원 추정: 첫 유효 벡터 길이, 실패 시 768(gemini text-embedding-004)
        def _len1(v):
            if isinstance(v, dict):
                for k in ("embedding","values","vector","data"):
                    if k in v: v=v[k]; break
            try:
                import numpy as np
                a = np.asarray(v)
                return (a.reshape(-1)).size
            except Exception:
                return 0
        dim = next(( _len1(v) for v in raw if _len1(v)>0 ), 768)

        # 4) 정규화해서 final_embs에 정확히 (dim,)로 채우기
        fixed = [ _coerce_vec(v, dim) for v in raw ]
        for (pos, _), emb in zip(new_docs, fixed):
            final_embs[pos] = emb

        LOG.info("[LOG] Embedded %d new docs", len(new_docs))

    emb_arr = np.array(final_embs, dtype="float32")
    np.save(vec_p, emb_arr)
    ids_p.write_text(json.dumps(doc_ids))
    bm25 = BM25Index(doc_ids, texts)
    emb_index = EmbeddingIndex(doc_ids, emb_arr)
    LOG.info("[LOG] Indexes built in %.2fs", time.time()-t0)
    return bm25, emb_index

def _hybrid_retrieve(question: str, bm25, emb_index, store: DocumentStore, top_k=10):
    hits = {}
    # BM25
    for did, s in bm25.search(question, k=top_k*4):
        hits[did] = max(hits.get(did, 0.0), float(s))
    # 임베딩
    qv = np.array(embed_query(question))
    for did, s in emb_index.search(qv, k=top_k*4):
        hits[did] = max(hits.get(did, 0.0), float(s))
    # 텍스트 동반
    out = [(did, sc, (store.get(did).text if store.get(did) else "")) for did, sc in hits.items()]
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_k]

def run_dragin(question: str, bm25, emb_index, store: DocumentStore, k: int = 5) -> Dict[str, Any]:
    start = time.time()
    # 최소 구현: 항상 검색(트리거는 methods.dragin 내부로 옮겨도 됨)
    hits = _hybrid_retrieve(question, bm25, emb_index, store, top_k=max(10,k*2))
    # 간단 rerank: 편집거리 기반(짧게)
    from ..retrieval.rerank import simple_rerank
    reranked = simple_rerank(question, hits, top_k=k)
    ctx_docs = [store.get(did).text for did,_,_ in reranked if store.get(did)][:3]
    context = "\n\n".join(ctx_docs)
    prompt = (
        "You are an extraction engine. Use ONLY the provided context. "
        "If the context does not contain the answer, reply exactly with: INSUFFICIENT_CONTEXT.\n"
        f"Question: {question}\nContext:\n{context}\nAnswer:"
    )
    ans = generate(prompt, temperature=0.0, max_output_tokens=256)
    return {"pred": ans, "retrieved": [{"doc_id": d, "score": s} for d,s,_ in reranked], "latency_ms": (time.time()-start)*1000}

def run_dota(question: str, bm25, emb_index, store: DocumentStore, k: int = 5) -> Dict[str, Any]:
    start = time.time()
    hits = _hybrid_retrieve(question, bm25, emb_index, store, top_k=max(10,k*2))
    from ..retrieval.rerank import simple_rerank
    reranked = simple_rerank(question, hits, top_k=k)
    ctx = "\n\n".join([store.get(d).text for d,_,_ in reranked if store.get(d)][:3])
    ans = generate(
        "You are an extraction engine. Use ONLY the provided context. "
        "If the context does not contain the answer, reply exactly with: INSUFFICIENT_CONTEXT.\n"
        f"Q: {question}\nContext:\n{ctx}\nA:",
        temperature=0.0, max_output_tokens=256
    )
    return {"pred": ans, "retrieved": [{"doc_id": d, "score": s} for d,s,_ in reranked], "latency_ms": (time.time()-start)*1000}

def run_graphrag(question: str, graph: GraphRAG, store: DocumentStore, k: int = 5) -> Dict[str, Any]:
    start = time.time()
    dids = graph.retrieve(question, hops=2, k=max(10,k*2))
    scored = [(d, 1.0, store.get(d).text if store.get(d) else "") for d in dids]
    from ..retrieval.rerank import simple_rerank
    reranked = simple_rerank(question, scored, top_k=k)
    ctx = "\n\n".join([store.get(d).text for d,_,_ in reranked if store.get(d)][:3])
    ans = generate(
        "You are an extraction engine. Use ONLY the provided context. "
        "If the context does not contain the answer, reply exactly with: INSUFFICIENT_CONTEXT.\n"
        f"Q: {question}\nGraph context:\n{ctx}\nA:", temperature=0.0, max_output_tokens=256
    )
    return {"pred": ans, "retrieved": [{"doc_id": d, "score": s} for d,s,_ in reranked], "latency_ms": (time.time()-start)*1000}

def run_hdrag(question: str, bm25, emb_index, store: DocumentStore, k: int = 5) -> Dict[str, Any]:
    start = time.time()
    # 컨텍스트에선 표-하이브리드 표현 사용
    hits = _hybrid_retrieve(question, bm25, emb_index, store, top_k=max(10,k*2))
    from ..retrieval.rerank import simple_rerank
    reranked = simple_rerank(question, hits, top_k=k)
    ctx_chunks = []
    for d,_,_ in reranked[:3]:
        doc = store.get(d)
        if not doc: continue
        hybrid = build_hybrid_representation({"doc_id": d, "text": doc.text, "tables": doc.metadata.get("tables", [])})
        ctx_chunks.append(hybrid)
    ctx = "\n\n".join(ctx_chunks)
    ans = generate(
        "You are an extraction engine. Use ONLY the provided context. "
        "If the context does not contain the answer, reply exactly with: INSUFFICIENT_CONTEXT.\n"
        f"Q: {question}\nContext:\n{ctx}\nA:", temperature=0.0, max_output_tokens=256
    )
    return {"pred": ans, "retrieved": [{"doc_id": d, "score": s} for d,s,_ in reranked], "latency_ms": (time.time()-start)*1000}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dragin","dota","graphrag","hdrag"], required=False, default="dota")
    ap.add_argument("--herb_root", required=True)
    ap.add_argument("--questions", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="process only first N questions")
    args = ap.parse_args()

    store, doc_ids, texts = load_corpus(args.herb_root)
    bm25, emb_index = build_indexes(doc_ids, texts)

    graph = None
    if args.method == "graphrag":
        LOG.info("[LOG] Building GraphRAG...")
        g0 = time.time()
        graph = GraphRAG()
        for d in store.all():
            graph.add_document(d.doc_id, d.text or "")
        LOG.info("[LOG] GraphRAG built in %.2fs", time.time()-g0)

    qpath = args.questions or os.path.join(args.herb_root, "data", "questions.jsonl")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    errlog = open("runs/errors.log", "a", encoding="utf-8")

    LOG.info("[LOG] Start processing questions from %s", qpath)
    n = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for line in open(qpath, "r", encoding="utf-8"):
            if not line.strip():
                continue
            q = json.loads(line)
            qid = str(q.get("qid") or q.get("id") or "")
            question = q.get("question") or q.get("query") or ""
            if not question:
                continue
            try:
                if args.method == "dragin":
                    res = run_dragin(question, bm25, emb_index, store, k=args.k)
                elif args.method == "dota":
                    res = run_dota(question, bm25, emb_index, store, k=args.k)
                elif args.method == "graphrag":
                    res = run_graphrag(question, graph, store, k=args.k)
                else:
                    res = run_hdrag(question, bm25, emb_index, store, k=args.k)
            except Exception as e:
                errlog.write(f"{qid}\t{args.method}\t{type(e).__name__}\t{e}\n")
                res = {"pred":"", "retrieved":[], "latency_ms":0.0, "error": str(e)}
            fout.write(json.dumps({"qid": qid, **res}, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    errlog.close()

if __name__ == "__main__":
    main()
