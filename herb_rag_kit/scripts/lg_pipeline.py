from __future__ import annotations
##__SRC_PATH_HACK__
import os, sys
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC=os.path.join(ROOT,'src')
sys.path.insert(0, SRC)

import os, re, json, time, argparse
from typing import TypedDict, Literal, List, Dict, Any, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
from tqdm import tqdm

# LangGraph
from langgraph.graph import StateGraph, START, END

# Neo4j 래퍼 (이미 패치한 파일)
from herb_rag_kit.graphdb.neo4j_store import Neo4jStore

# LLM(Gemini)
import google.generativeai as genai

# 임베딩 폴백
from sentence_transformers import SentenceTransformer
import math
# ----------------------- 설정 -----------------------

@dataclass
class Config:
    gemini_model: str = os.getenv("GEMINI_MODEL", "models/gemini-2.5-pro")
    emb_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))
    chunk_chars: int = int(os.getenv("LG_CHUNK_CHARS", "1200"))
    chunk_overlap: int = int(os.getenv("LG_CHUNK_OVERLAP", "120"))
    max_chunks_per_doc: int = int(os.getenv("LG_MAX_CHUNKS", "8"))
    redact: bool = os.getenv("LG_REDACT", "1") not in ("0", "false", "False")
    # Cleaning / consistency / bias mitigation
    conf_threshold: float = float(os.getenv("LG_CONF_THRESHOLD", "0.6"))
    enable_consistency: bool = os.getenv("LG_CONSISTENCY", "0") in ("1","true","True")
    max_objects_per_sp: int = int(os.getenv("LG_MAX_OBJECTS_PER_SP", "5"))
    max_per_entity: int = int(os.getenv("LG_MAX_PER_ENTITY", "200"))

cfg = Config()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("GEMINI_API_KEY not set")
genai.configure(api_key=api_key)

# ----------------------- 유틸 -----------------------

NSFW_PAT = re.compile(
    r"(?i)\b(anal|bdsm|blowjob|boob|camgirl|cum|deepthroat|erotic|explicit sex|fetish|fuck|hentai|nsfw|nude|orgasm|porn|rape|sex|sexual|sext|sexting|strip|xxx)\b"
)

def redact_nsfw(text: str) -> str:
    if not cfg.redact:
        return text
    out = []
    for ln in text.splitlines():
        if NSFW_PAT.search(ln):
            continue
        out.append(ln)
    return "\n".join(out)

def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text: return []
    chunks, i, n = [], 0, len(text)
    while i < n:
        j = min(n, i + chunk_chars)
        chunks.append(text[i:j])
        if j >= n: break
        i = max(j - overlap, j)
    return chunks

# ----------------------- LLM 호출 -----------------------

PROMPT_TRIPLES = """You are an information extraction model.
Extract factual SPO triples explicitly stated in the text.
Focus on business/organization facts only (people, orgs, roles, products, dates, locations).
Skip any sensitive/sexual/off-topic content if present.
Predicates must be lower_snake_case (e.g., works_at, founded_by, located_in, reports_to, partners_with, acquired, uses, builds, mentions).
Return ONLY JSON of this schema:
[
  {"s": "...", "p": "...", "o": "...", "s_type": "PERSON|ORG|LOC|PRODUCT|EVENT|WORK|THING", "o_type": "...", "conf": 0.0-1.0}
]
No extra text, no code fences.
Text:
\"\"\"{text}\"\"\""""

def extract_text_from_response(resp) -> str:
    try:
        if getattr(resp, "text", None):
            return resp.text
    except Exception:
        pass
    try:
        for c in getattr(resp, "candidates", []) or []:
            for part in getattr(c, "content", {}).parts or []:
                if getattr(part, "text", None):
                    return part.text
    except Exception:
        pass
    return ""

def gemini_generate_json(model: str, prompt: str) -> List[Dict[str,Any]]:
    m = genai.GenerativeModel(model_name=model, generation_config={
        "temperature": 0.0, "max_output_tokens": 2048, "response_mime_type": "application/json"
    })
    resp = m.generate_content(prompt)
    out = extract_text_from_response(resp).strip()
    if not out:
        return []
    try:
        data = json.loads(out)
    except Exception:
        l, r = out.find("["), out.rfind("]")
        data = json.loads(out[l:r+1]) if l!=-1 and r!=-1 else []
    if not isinstance(data, list):
        return []
    return data

def text2cypher_with_gemini(question: str) -> str:
    system = (
        "You are a Cypher assistant. Only output a single read-only Cypher query using MATCH/RETURN.\n"
        "Do NOT use CREATE/MERGE/SET/DELETE.\n"
        "Schema: Nodes => (Document {id,title,text,url,embedding}), (Entity {name,type,embedding}); "
        "Rels => (Entity)-[:RELATES {predicate,confidence}]->(Entity), (Entity)-[:APPEARS_IN]->(Document). "
        "Avoid APPEARS_IN unless the user explicitly asks about documents; prefer RELATES patterns."
    )
    prompt = f"{system}\n\nQ: {question}\nCypher:"
    try:
        m = genai.GenerativeModel(model_name=cfg.gemini_model, generation_config={"temperature":0.0,"max_output_tokens":256})
        resp = m.generate_content(prompt)
        out = extract_text_from_response(resp).strip()
        out = out.strip("`").strip()
        if out.lower().startswith("cypher"):
            out = out.split("\n",1)[-1]
        if not out.lower().startswith("match"):
            raise ValueError("not read-only")
        return out
    except Exception:
        return "MATCH (d:Document) RETURN d.id AS id, d.title AS title LIMIT 5"

def try_gemini_embed(q: str, dim: int) -> Optional[List[float]]:
    # 사용자가 herb_rag_kit.llm.gemini_client.embed_query 를 갖고 있으면 우선 사용
    try:
        from herb_rag_kit.llm.gemini_client import embed_query
        v = embed_query(q)
        v = v.tolist() if hasattr(v,"tolist") else v
        v = [float(x) for x in v]
        if len(v) != dim: return None
        if not all(math.isfinite(x) for x in v): return None
        if sum(x*x for x in v) <= 0.0: return None
        return v
    except Exception:
        return None

def try_st_embed(q: str, dim: int) -> Optional[List[float]]:
    try:
        model = SentenceTransformer(os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2"))
        v = model.encode([q], normalize_embeddings=True)[0]
        if v.shape[0] != dim: return None
        return v.astype("float32").tolist()
    except Exception:
        return None

def get_query_vec(q: str, dim: int) -> Optional[List[float]]:
    v = try_gemini_embed(q, dim)
    if v is not None: return v
    v = try_st_embed(q, dim)
    if v is not None: return v
    return None

# ----------------------- 상태 & 노드 -----------------------

class State(TypedDict, total=False):
    task: Literal["extract","query"]
    # 공통
    stats: Dict[str, Any]
    # 추출 경로
    doc_id: str
    text: str
    chunks: List[str]
    triples: List[Dict[str,Any]]
    extracted: int
    # 질의 경로
    question: str
    cypher: str
    graph_rows: List[Dict[str,Any]]
    hybrid_rows: List[Dict[str,Any]]
    k: int

store = Neo4jStore()

# Ingest용 Cypher (Community-safe)
CYPHER_INGEST = """
UNWIND $rows AS t
MERGE (s:Entity {key: t.s_key})
  ON CREATE SET s.name=t.s, s.type=t.s_type
SET s.name=coalesce(s.name,t.s), s.type=coalesce(s.type,t.s_type)
MERGE (o:Entity {key: t.o_key})
  ON CREATE SET o.name=t.o, o.type=t.o_type
SET o.name=coalesce(o.name,t.o), o.type=coalesce(o.type,t.o_type)
MERGE (s)-[r:RELATES {predicate:t.p}]->(o)
  ON CREATE SET r.firstSeen=timestamp(), r.confidence=coalesce(t.conf,1.0)
WITH t,s,o,r
SET r.confidence = CASE
  WHEN r.confidence IS NULL THEN coalesce(t.conf,1.0)
  WHEN coalesce(t.conf,0.0) > r.confidence THEN coalesce(t.conf,1.0)
  ELSE r.confidence
END
WITH t,s,o
OPTIONAL MATCH (d:Document {id: t.doc_id})
FOREACH (_ IN CASE WHEN d IS NULL THEN [] ELSE [1] END |
  MERGE (s)-[:APPEARS_IN]->(d)
  MERGE (o)-[:APPEARS_IN]->(d)
)
"""

def route(state: State) -> Literal["extract_branch","query_branch"]:
    return "extract_branch" if state.get("task") == "extract" else "query_branch"

def ensure_schema(state: State) -> State:
    store.create_schema()
    st = state.get("stats", {})
    st["schema_ok"] = True
    state["stats"] = st
    return state

# ---- 추출 브랜치 ----
def redact_and_chunk(state: State) -> State:
    txt = state["text"]
    if cfg.redact:
        txt = redact_nsfw(txt)
    chunks = chunk_text(txt, cfg.chunk_chars, cfg.chunk_overlap)[:cfg.max_chunks_per_doc]
    state["chunks"] = chunks
    return state

def llm_extract(state: State) -> State:
    doc_id = state["doc_id"]
    triples_all: List[Dict[str,Any]] = []
    seen = set()
    for ch in state.get("chunks", []):
        try:
            raw = gemini_generate_json(cfg.gemini_model, PROMPT_TRIPLES.format(text=ch))
        except Exception:
            raw = []
        for t in raw:
            s = (t.get("s") or "").strip(); p = (t.get("p") or "").strip(); o = (t.get("o") or "").strip()
            if not (s and p and o): continue
            s_type = (t.get("s_type") or "THING").strip().upper()
            o_type = (t.get("o_type") or "THING").strip().upper()
            conf = float(t.get("conf", 0.7)) if str(t.get("conf","")).strip()!="" else 0.7
            key = (s,p,o,s_type,o_type,doc_id)
            if key in seen: 
                continue
            seen.add(key)
            triples_all.append({
                "s": s, "p": p, "o": o,
                "s_type": s_type, "o_type": o_type,
                "s_key": f"{s}|{s_type}", "o_key": f"{o}|{o_type}",
                "doc_id": doc_id, "conf": conf
            })
    state["triples"] = triples_all
    state["extracted"] = len(triples_all)
    return state

def _canon_text(x: str) -> str:
    x = (x or "").strip()
    # collapse spaces
    x = re.sub(r"\s+", " ", x)
    return x

def _canon_pred(p: str) -> str:
    p = (p or "").strip()
    p = p.replace(" ", "_").replace("-", "_")
    p = re.sub(r"[^a-zA-Z0-9_]+", "", p)
    return p.lower()

def normalize_triples(state: State) -> State:
    rows = state.get("triples", [])
    out = []
    seen = set()
    removed_lowconf = 0
    removed_long = 0
    for t in rows:
        s = _canon_text(t.get("s",""))
        o = _canon_text(t.get("o",""))
        p = _canon_pred(t.get("p",""))
        s_type = (t.get("s_type") or "THING").strip().upper()
        o_type = (t.get("o_type") or "THING").strip().upper()
        conf = float(t.get("conf", 0.7)) if str(t.get("conf",""))!="" else 0.7

        if conf < cfg.conf_threshold:
            removed_lowconf += 1
            continue
        # Guard absurd string length to reduce noise
        if len(s) > 300 or len(o) > 300:
            removed_long += 1
            continue

        rec = {
            "s": s, "o": o, "p": p,
            "s_type": s_type, "o_type": o_type,
            "s_key": f"{s}|{s_type}",
            "o_key": f"{o}|{o_type}",
            "doc_id": (t.get("doc_id") or ""),
            "conf": conf,
        }
        key = (rec["s_key"], rec["p"], rec["o_key"], rec["doc_id"]) 
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)

    st = state.get("stats", {})
    st["normalize_in"] = len(rows)
    st["normalize_out"] = len(out)
    st["removed_lowconf"] = removed_lowconf
    st["removed_long"] = removed_long
    state["stats"] = st
    state["triples"] = out
    state["extracted"] = len(out)
    return state

def consistency_filter(state: State) -> State:
    if not cfg.enable_consistency:
        return state
    rows = state.get("triples", [])
    by_sp: dict[tuple, list] = defaultdict(list)
    for t in rows:
        by_sp[(t["s_key"], t["p"])].append(t)
    kept = []
    removed = 0
    for (sk, p), group in by_sp.items():
        # Prefer top objects by (count, avg_conf)
        o_counts = Counter([t["o_key"] for t in group])
        # rank objects
        o_to_conf = defaultdict(list)
        for t in group:
            o_to_conf[t["o_key"]].append(float(t.get("conf", 0.7)))
        ranked = sorted(o_counts.items(), key=lambda kv: (kv[1], sum(o_to_conf[kv[0]])/len(o_to_conf[kv[0]])), reverse=True)
        allowed = set([o for o,_ in ranked[:max(1, cfg.max_objects_per_sp)]])
        for t in group:
            if t["o_key"] in allowed:
                kept.append(t)
            else:
                removed += 1
    st = state.get("stats", {})
    st["consistency_removed"] = removed
    state["stats"] = st
    state["triples"] = kept
    state["extracted"] = len(kept)
    return state

def bias_mitigate(state: State) -> State:
    rows = state.get("triples", [])
    # Cap per-entity outgoing edges to reduce hub dominance
    by_entity: dict[str, list] = defaultdict(list)
    for t in rows:
        by_entity[t["s_key"]].append(t)
    capped = []
    dropped = 0
    for sk, group in by_entity.items():
        # sort by (confidence desc, rare predicate first to increase diversity)
        pred_counts = Counter([t["p"] for t in group])
        group_sorted = sorted(group, key=lambda t: (float(t.get("conf",0.7)), -pred_counts[t["p"]]), reverse=True)
        keep_n = min(len(group_sorted), max(0, cfg.max_per_entity))
        capped.extend(group_sorted[:keep_n])
        dropped += max(0, len(group_sorted) - keep_n)
    st = state.get("stats", {})
    st["bias_dropped"] = dropped
    state["stats"] = st
    state["triples"] = capped
    state["extracted"] = len(capped)
    return state

def ingest_triples(state: State) -> State:
    rows = state.get("triples", [])
    if rows:
        store.run_cypher(CYPHER_INGEST, {"rows": rows})
    st = state.get("stats", {})
    st["last_doc_extracted"] = len(rows)
    state["stats"] = st
    return state

# ---- 질의 브랜치 ----
def t2c(state: State) -> State:
    cy = text2cypher_with_gemini(state["question"])
    state["cypher"] = cy
    return state

def run_cypher(state: State) -> State:
    rows = store.run_cypher(state["cypher"])
    state["graph_rows"] = rows
    return state

def run_knn(state: State) -> State:
    k = int(state.get("k", 5))
    qvec = get_query_vec(state["question"], store.cfg.emb_dim)
    if qvec is None:
        state["hybrid_rows"] = []
        return state
    hyb = store.knn(qvec, k=k)
    state["hybrid_rows"] = hyb
    return state

# ----------------------- 그래프 구성 -----------------------

g = StateGraph(State)
g.add_node("ensure_schema", ensure_schema)
g.add_node("redact_and_chunk", redact_and_chunk)
g.add_node("llm_extract", llm_extract)
g.add_node("normalize_triples", normalize_triples)
g.add_node("consistency_filter", consistency_filter)
g.add_node("bias_mitigate", bias_mitigate)
g.add_node("ingest_triples", ingest_triples)

g.add_node("t2c", t2c)
g.add_node("run_cypher", run_cypher)
g.add_node("run_knn", run_knn)

# 라우팅
g.add_conditional_edges(START, route, {
    "extract_branch": "ensure_schema",
    "query_branch": "t2c",
})

# extract branch
g.add_edge("ensure_schema", "redact_and_chunk")
g.add_edge("redact_and_chunk", "llm_extract")
g.add_edge("llm_extract", "normalize_triples")
g.add_edge("normalize_triples", "consistency_filter")
g.add_edge("consistency_filter", "bias_mitigate")
g.add_edge("bias_mitigate", "ingest_triples")
g.add_edge("ingest_triples", END)

# query branch
g.add_edge("t2c", "run_cypher")
g.add_edge("run_cypher", "run_knn")
g.add_edge("run_knn", END)

app = g.compile()

# ----------------------- CLI -----------------------

def load_docstore(jsonl_path: str, limit: int = 0):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            d = json.loads(line)
            if "id" not in d or "text" not in d:
                continue
            yield {"id": d["id"], "text": d["text"], "title": d.get("title")}
            if limit and i >= limit:
                break

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="LLM 추출→클린/일관성/바이어스완화→Neo4j")
    ex.add_argument("--docstore", default="./indexes/herb/docstore.jsonl")
    ex.add_argument("--limit", type=int, default=0, help="문서 수 제한(0=전체)")
    ex.add_argument("--max-chunks", type=int, default=cfg.max_chunks_per_doc)
    ex.add_argument("--conf-threshold", type=float, default=cfg.conf_threshold)
    ex.add_argument("--consistency-llm", action="store_true", help="(옵션) 일관성 필터 활성화")
    ex.add_argument("--max-objects-per-sp", type=int, default=cfg.max_objects_per_sp)
    ex.add_argument("--max-per-entity", type=int, default=cfg.max_per_entity)

    qy = sub.add_parser("query", help="GraphRAG 질의(하이브리드)")
    qy.add_argument("--q", required=True)
    qy.add_argument("--k", type=int, default=5)

    args = ap.parse_args()

    if args.cmd == "extract":
        # override runtime config
        cfg.max_chunks_per_doc = int(args.max_chunks)
        cfg.conf_threshold = float(args.conf_threshold)
        cfg.enable_consistency = bool(args.consistency_llm)
        cfg.max_objects_per_sp = int(args.max_objects_per_sp)
        cfg.max_per_entity = int(args.max_per_entity)
        total = 0
        for doc in tqdm(load_docstore(args.docstore, args.limit), desc="extract+ingest"):
            # per-doc 실행 (청크는 노드 내부에서 처리)
            state = app.invoke({
                "task":"extract",
                "doc_id": doc["id"],
                "text": doc["text"],
                "stats": {}
            })
            total += state.get("extracted", 0)
        print(json.dumps({"extracted_total": total}, ensure_ascii=False, indent=2))

    elif args.cmd == "query":
        st = app.invoke({
            "task":"query",
            "question": args.q,
            "k": args.k,
            "stats": {}
        })
        out = {
            "question": args.q,
            "cypher": st.get("cypher"),
            "graph_rows": st.get("graph_rows", [])[:args.k],
            "hybrid_rows": st.get("hybrid_rows", []),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
