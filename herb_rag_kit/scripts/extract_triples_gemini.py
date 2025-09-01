import os, re, json, argparse, time, pathlib
from typing import List, Dict, Any, Iterable, Optional
import google.generativeai as genai

def load_docstore(jsonl_path: str, limit: int = 0) -> Iterable[Dict[str, Any]]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            d = json.loads(line)
            if "id" not in d or "text" not in d:
                continue
            yield {"id": d["id"], "title": d.get("title"), "text": d["text"]}
            if limit and i >= limit:
                break

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

PROMPT = """You are an information extraction model.
Extract factual SPO triples explicitly stated in the text.
Predicates must be lower_snake_case (e.g., works_at, founded_by, located_in).
Guess coarse types if obvious: PERSON, ORG, LOC, PRODUCT, EVENT, WORK, THING.
Return ONLY JSON of this schema:
[
  {"s": "...", "p": "...", "o": "...", "s_type": "PERSON|ORG|LOC|PRODUCT|EVENT|WORK|THING", "o_type": "PERSON|ORG|LOC|PRODUCT|EVENT|WORK|THING", "conf": 0.0-1.0}
]
No extra text, no code fences.
Text:
\"\"\"{text}\"\"\""""

def extract_text_from_response(resp) -> str:
    # 1) resp.text (SDK가 지원할 때)
    try:
        if getattr(resp, "text", None):
            return resp.text
    except Exception:
        pass
    # 2) candidates -> parts -> text
    try:
        for c in getattr(resp, "candidates", []) or []:
            for part in getattr(c, "content", {}).parts or []:
                if getattr(part, "text", None):
                    return part.text
    except Exception:
        pass
    return ""

def call_gemini(model_name: str, text: str, retries: int = 3, sleep_s: float = 1.5, verbose: bool=False) -> List[Dict[str, Any]]:
    last_err = None
    for t in range(retries):
        try:
            m = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 2048,
                    "response_mime_type": "application/json"
                },
                safety_settings=[  # 널널하게
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUAL_CONTENT", "threshold": "BLOCK_NONE"},
                ],
            )
            resp = m.generate_content(PROMPT.format(text=text))
            out = extract_text_from_response(resp).strip()
            if not out:
                if verbose: print("[WARN] empty response; retrying…")
                raise RuntimeError("empty_response")

            # JSON robust parsing
            triples = None
            try:
                triples = json.loads(out)
            except Exception:
                l, r = out.find("["), out.rfind("]")
                if l != -1 and r != -1:
                    triples = json.loads(out[l:r+1])
            if not isinstance(triples, list):
                triples = []

            clean = []
            for it in triples:
                if not isinstance(it, dict): continue
                s = (it.get("s") or "").strip()
                p = (it.get("p") or "").strip()
                o = (it.get("o") or "").strip()
                if not (s and p and o): continue
                s_type = (it.get("s_type") or "THING").strip().upper()
                o_type = (it.get("o_type") or "THING").strip().upper()
                try: conf = float(it.get("conf", 0.7))
                except: conf = 0.7
                clean.append({"s":s,"p":p,"o":o,"s_type":s_type,"o_type":o_type,"conf":conf})
            return clean

        except Exception as e:
            last_err = e
            if verbose: print(f"[ERR] call_gemini({model_name}) attempt {t+1}/{retries} -> {e}")
            time.sleep(sleep_s * (t+1))
    if verbose and last_err: print("[ERR] final:", last_err)
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./indexes/herb/docstore.jsonl", help="문서 JSONL (id,text,title,...)")
    ap.add_argument("--out", default="./data/triples.llm.jsonl", help="추출 결과 JSONL")
    ap.add_argument("--model", default=os.getenv("GEMINI_MODEL","gemini-2.5-pro"))
    ap.add_argument("--chunk_chars", type=int, default=4000)
    ap.add_argument("--overlap", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0, help="문서 수 제한(0=전체)")
    ap.add_argument("--max_chunks_per_doc", type=int, default=8, help="문서 당 최대 처리 청크")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set in environment or .env")
    genai.configure(api_key=api_key)

    pathlib.Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    failures_dir = pathlib.Path("./data/.failures"); failures_dir.mkdir(parents=True, exist_ok=True)

    seen_keys = set()
    total = 0
    doc_cnt = 0
    with open(args.out, "w", encoding="utf-8") as wf:
        for doc in load_docstore(args.input, limit=args.limit):
            doc_cnt += 1
            doc_id = doc["id"]
            chunks = chunk_text(doc["text"], args.chunk_chars, args.overlap)[:args.max_chunks_per_doc]
            if args.verbose: print(f"[DOC] {doc_cnt} {doc_id} chunks={len(chunks)}")
            for ci, ch in enumerate(chunks):
                triples = call_gemini(args.model, ch, verbose=args.verbose)
                if not triples:
                    # 실패 사례 저장(디버깅용)
                    with open(failures_dir / f"{doc_id.replace('/','_')}.{ci}.json", "w", encoding="utf-8") as ff:
                        ff.write(json.dumps({"doc_id":doc_id,"chunk_idx":ci,"text_sample":ch[:500]}, ensure_ascii=False))
                for t in triples:
                    key = (t["s"], t["p"], t["o"], t["s_type"], t["o_type"], doc_id)
                    if key in seen_keys: 
                        continue
                    seen_keys.add(key)
                    t["doc_id"] = doc_id
                    wf.write(json.dumps(t, ensure_ascii=False) + "\n")
                    total += 1
    print(f"[OK] docs={doc_cnt}, extracted triples={total} → {args.out}")

if __name__ == "__main__":
    main()
