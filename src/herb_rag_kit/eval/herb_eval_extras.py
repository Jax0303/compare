
import json, argparse, sys, os, math, statistics
from typing import List, Dict, Any, Tuple, Optional, Iterable
from datetime import datetime, timezone

ISO_FORMATS = [
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]

def parse_time(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    for fmt in ISO_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            continue
    return None

def read_json_or_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(2)
        f.seek(0)
        if first.strip().startswith("["):
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]

def normalize_answer(a: Any) -> List[str]:
    if a is None:
        return []
    if isinstance(a, str):
        return [a.strip()]
    if isinstance(a, (list, tuple, set)):
        return [str(x).strip() for x in a if str(x).strip()]
    return [str(a).strip()]

def em(pred: List[str], gold: List[str]) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    pset = {p.lower().strip() for p in pred}
    gset = {g.lower().strip() for g in gold}
    return 1.0 if pset & gset else 0.0

def f1_set(pred: List[str], gold: List[str]) -> float:
    pset = {p.lower().strip() for p in pred if p}
    gset = {g.lower().strip() for g in gold if g}
    if not pset and not gset:
        return 1.0
    if not pset or not gset:
        return 0.0
    tp = len(pset & gset)
    fp = len(pset - gset)
    fn = len(gset - pset)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def hit_at_k(retrieved, gold_doc_ids, k: int) -> float:
    tops = [r.get("doc_id") for r in (retrieved or [])[:k]]
    gset = set(gold_doc_ids or [])
    return 1.0 if any(doc_id in gset for doc_id in tops) else 0.0

def fresh_at_k(retrieved, gold_doc_ids, k: int, t0: Optional[datetime]) -> float:
    if t0 is None:
        return float("nan")
    gset = set(gold_doc_ids or [])
    for r in (retrieved or [])[:k]:
        ts = parse_time(r.get("timestamp"))
        if ts and ts >= t0 and r.get("doc_id") in gset:
            return 1.0
    return 0.0

def latency_stats(latencies):
    latencies = [x for x in (latencies or []) if isinstance(x, (int, float))]
    if not latencies:
        return {"mean_ms": float("nan"), "p50_ms": float("nan"), "p95_ms": float("nan")}
    latencies_sorted = sorted(latencies)
    def pct(p):
        idx = max(0, min(len(latencies_sorted)-1, int(round((p/100.0) * (len(latencies_sorted)-1)))))
        return latencies_sorted[idx]
    return {
        "mean_ms": sum(latencies)/len(latencies),
        "p50_ms": pct(50),
        "p95_ms": pct(95),
    }

def build_index_by_qid(items, key_qid: str="qid"):
    out = {}
    for it in (items or []):
        qid = str(it.get(key_qid))
        if qid:
            out[qid] = it
    return out

def extract_fields_pred(item):
    qid = str(item.get("qid"))
    p = item.get("pred") or item.get("predicted_answer") or item.get("answer")
    pred = normalize_answer(p)
    retrieved = item.get("retrieved") or item.get("docs") or []
    latency = item.get("latency_ms") or item.get("latency")
    return qid, pred, retrieved, latency

def extract_fields_gold(item):
    qid = str(item.get("qid"))
    gold_answers = item.get("answers") or item.get("gold_answers") or item.get("answer")
    gold = normalize_answer(gold_answers)
    gold_docs = item.get("gold_docs") or item.get("evidence_docs") or []
    gold_ids = [d.get("doc_id") if isinstance(d, dict) else d for d in gold_docs]
    return qid, gold, gold_ids

def evaluate_single_run(pred_items, gold_items, ks, t0):
    gold_by_qid = build_index_by_qid(gold_items)
    em_scores, f1_scores = [], []
    hits = {k: [] for k in ks}
    fresh_hits = {k: [] for k in ks}
    latencies = []
    per_q = {}

    for it in pred_items:
        qid, pred, retrieved, latency = extract_fields_pred(it)
        gold_item = gold_by_qid.get(qid)
        if not gold_item:
            continue
        _, gold_ans, gold_ids = extract_fields_gold(gold_item)
        em_score = em(pred, gold_ans)
        f1_score = f1_set(pred, gold_ans)
        em_scores.append(em_score)
        f1_scores.append(f1_score)
        for k in ks:
            hits[k].append(hit_at_k(retrieved, gold_ids, k))
            fresh_hits[k].append(fresh_at_k(retrieved, gold_ids, k, t0))
        if latency is not None:
            try:
                latencies.append(float(latency))
            except:
                pass
        per_q[qid] = {
            "em": em_score,
            "f1": f1_score,
            **{f"hit@{k}": hits[k][-1] for k in ks},
            **({f"fresh@{k}": fresh_hits[k][-1] for k in ks} if t0 else {}),
            "latency_ms": latency,
        }

    res = {
        "EM": sum(em_scores)/len(em_scores) if em_scores else float("nan"),
        "F1": sum(f1_scores)/len(f1_scores) if f1_scores else float("nan"),
        "Hit@K": {str(k): sum(h)/len(h) if h else float("nan") for k, h in hits.items()},
        **({"Fresh@K": {str(k): sum(h)/len(h) if h else float("nan") for k, h in fresh_hits.items()}} if t0 else {}),
        "Latency": latency_stats(latencies),
        "N": len(per_q),
        "PerQuestion": per_q,
    }
    return res

def compute_tti(runs, gold_items, threshold: float = 0.5, metric: str = "hit@1"):
    gold_by_qid = build_index_by_qid(gold_items)
    tti = {}
    for round_idx, preds in enumerate(runs, start=1):
        pred_by_q = build_index_by_qid(preds)
        for qid, gold_item in gold_by_qid.items():
            pitem = pred_by_q.get(qid)
            if not pitem:
                continue
            qid2, pred, retrieved, _ = extract_fields_pred(pitem)
            _, gold_ans, gold_ids = extract_fields_gold(gold_item)
            m = metric.lower()
            if m.startswith("hit@"):
                try:
                    k = int(m.split("@")[1])
                except:
                    k = 1
                val = hit_at_k(retrieved, gold_ids, k)
            elif m == "em":
                val = em(pred, gold_ans)
            elif m == "f1":
                val = f1_set(pred, gold_ans)
            else:
                continue
            if val >= threshold and qid not in tti:
                tti[qid] = round_idx
    return tti

def cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True, help="One or more prediction files (JSON or JSONL). If more than one, TTI will be computed across rounds in given order.")
    ap.add_argument("--gold", required=True, help="Gold file (JSON or JSONL).")
    ap.add_argument("--t0", default=None, help="ISO time for Fresh@K (e.g., 2025-01-01T00:00:00Z). Omit to skip Fresh@K.")
    ap.add_argument("--k", nargs="+", type=int, default=[1,3,5,10], help="K values for Hit@K/Fresh@K.")
    ap.add_argument("--tti_metric", default="hit@1", choices=["hit@1", "em", "f1"], help="Metric used for TTI.")
    ap.add_argument("--tti_threshold", type=float, default=0.5, help="Threshold for TTI metric.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    args = ap.parse_args()

    t0 = parse_time(args.t0) if args.t0 else None
    gold_items = read_json_or_jsonl(args.gold)
    pred_runs = [read_json_or_jsonl(p) for p in args.pred]

    results = {}
    if len(pred_runs) == 1:
        results["run1"] = evaluate_single_run(pred_runs[0], gold_items, args.k, t0)
    else:
        for i, pr in enumerate(pred_runs, start=1):
            results[f"run{i}"] = evaluate_single_run(pr, gold_items, args.k, t0)
        tti = compute_tti(pred_runs, gold_items, threshold=args.tti_threshold, metric=args.tti_metric)
        results["TTI"] = {
            "metric": args.tti_metric,
            "threshold": args.tti_threshold,
            "per_question_round": tti,
            "mean_rounds_to_threshold": (sum(tti.values())/len(tti)) if tti else float("inf")
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    cli()
