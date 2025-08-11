from __future__ import annotations
import argparse, json, os, pathlib, re
from typing import Any, Dict, List
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "Salesforce/HERB"   # dataset repo
REPO_TYPE = "dataset"

def list_files(prefix: str) -> List[str]:
    api = HfApi()
    return [f for f in api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE) if f.startswith(prefix) and f.endswith(".json")]

def load_json(path_in_repo: str) -> Any:
    fp = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=path_in_repo)
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

def pick_text(d: Dict[str, Any]) -> str:
    # Slack-like
    if isinstance(d.get("Message"), dict) and isinstance(d["Message"].get("User"), dict):
        t = d["Message"]["User"].get("text")
        if isinstance(t, str): return t
    # Common fields
    for k in ("text", "content", "body", "message", "notes", "description", "title"):
        v = d.get(k)
        if isinstance(v, str) and v.strip(): return v
    # Fallback: concatenate string leaves
    parts = []
    def collect(x):
        if isinstance(x, dict):
            for v in x.values(): collect(v)
        elif isinstance(x, list):
            for v in x: collect(v)
        elif isinstance(x, str):
            parts.append(x)
    collect(d)
    return "\n".join(parts)[:8000]

def pick_timestamp(d: Dict[str, Any]) -> str | None:
    for path in (("timestamp",), ("time",), ("created_at",), ("Message","User","timestamp")):
        cur = d
        ok=True
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok=False; break
        if ok and isinstance(cur, str): return cur
    return None

def find_question_items(prod: Dict[str, Any]) -> List[Dict[str, Any]]:
    # collect dicts having a 'question' field anywhere under product root arrays
    qitems = []
    for k, v in prod.items():
        if isinstance(v, list):
            for it in v:
                if isinstance(it, dict) and any(key.lower()=="question" for key in it.keys()):
                    qitems.append(it)
    return qitems

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="e.g., ~/HERB_repo/data")
    args = ap.parse_args()

    out_root = pathlib.Path(os.path.expanduser(args.out_dir))
    (out_root / "corpus").mkdir(parents=True, exist_ok=True)

    # Load metadata (employees/customers) for enrichment if needed
    meta = {}
    for mf in list_files("metadata/"):
        try:
            meta[mf] = load_json(mf)
        except Exception:
            pass  # keep best-effort

    qid = 0
    wrote_docs = set()
    qf = open(out_root / "questions.jsonl", "w", encoding="utf-8")
    gf = open(out_root / "gold.jsonl", "w", encoding="utf-8")

    prod_files = list_files("products/")
    for pf in prod_files:
        product_name = pathlib.Path(pf).stem
        prod = load_json(pf)

        # ---- 1) flatten artifacts into corpus docs ----
        for k, v in prod.items():
            if not isinstance(v, list): continue
            # treat arrays of dicts as artifact collections
            if not v or not isinstance(v[0], dict): continue
            typ = k  # e.g., slack, documents, urls, pull_requests, meeting_transcripts, etc.
            for i, item in enumerate(v):
                # build doc_id
                did = (item.get("id") or item.get("doc_id") or item.get("utterranceID") or f"{product_name}-{typ}-{i:06d}")
                did = f"{product_name}::{typ}::{did}"
                if did in wrote_docs: continue

                text = pick_text(item)
                ts = pick_timestamp(item)
                doc = {
                    "doc_id": did,
                    "text": text,
                    "timestamp": ts,
                    "metadata": {
                        "product": product_name,
                        "type": typ
                    }
                }
                with open(out_root / "corpus" / f"{re.sub(r'[^A-Za-z0-9:_\\-]', '_', did)}.json", "w", encoding="utf-8") as fdoc:
                    json.dump(doc, fdoc, ensure_ascii=False)
                wrote_docs.add(did)

        # ---- 2) extract questions (any item with 'question' key) ----
        for qi in find_question_items(prod):
            q_text = qi.get("question") or ""
            answers = qi.get("answers") or qi.get("answer") or []
            if isinstance(answers, str): answers = [answers]
            # evidence IDs if present
            gold_docs = []
            for cand_key in ("evidence_ids", "doc_ids", "support_docs", "evidence"):
                if isinstance(qi.get(cand_key), list):
                    for did in qi[cand_key]:
                        if isinstance(did, str):
                            gold_docs.append({"doc_id": did, "timestamp": None})
                elif isinstance(qi.get(cand_key), dict):
                    for did in qi[cand_key].values():
                        if isinstance(did, str):
                            gold_docs.append({"doc_id": did, "timestamp": None})
            qid += 1
            qf.write(json.dumps({"qid": str(qid), "question": q_text}, ensure_ascii=False) + "\n")
            gf.write(json.dumps({"qid": str(qid), "answers": answers, "gold_docs": gold_docs}, ensure_ascii=False) + "\n")

    qf.close(); gf.close()
    print(f"OK: wrote {qid} questions and {len(wrote_docs)} documents under {out_root}")

if __name__ == "__main__":
    main()
