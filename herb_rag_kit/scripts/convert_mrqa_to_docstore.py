from __future__ import annotations
import os, sys, json, argparse, glob, hashlib
from pathlib import Path
from typing import Iterable, Dict, Any, List


def content_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def iter_json_lines(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        # Heuristic: if starts with '{' it's probably a single JSON (MRQA original), else JSONL
        if first == '{':
            try:
                obj = json.load(f)
            except Exception:
                return
            yield from expand_mrqa_object(obj)
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                yield from expand_mrqa_object(obj)


def expand_mrqa_object(obj: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    # Cases:
    # 1) JSONL with {"context": "...", ...}
    if isinstance(obj, dict) and "context" in obj:
        ctx = (obj.get("context") or "").strip()
        if ctx:
            yield {"text": ctx, "title": obj.get("title") or obj.get("source")}
        return

    # 2) SQuAD-like: {"data": [{"title":..., "paragraphs": [{"context":..., "qas": [...]}, ...]}]}
    data = obj.get("data") if isinstance(obj, dict) else None
    if isinstance(data, list):
        for art in data:
            title = art.get("title") if isinstance(art, dict) else None
            paragraphs = art.get("paragraphs", []) if isinstance(art, dict) else []
            for para in paragraphs:
                ctx = (para.get("context") or "").strip()
                if ctx:
                    yield {"text": ctx, "title": title}
        return

    # 3) Direct paragraphs list
    if isinstance(obj, dict) and "paragraphs" in obj:
        title = obj.get("title")
        for para in obj.get("paragraphs") or []:
            ctx = (para.get("context") or "").strip()
            if ctx:
                yield {"text": ctx, "title": title}
        return


def iter_inputs(input_path: str) -> Iterable[Dict[str, Any]]:
    p = Path(input_path)
    if p.is_file():
        yield from iter_json_lines(str(p))
    else:
        for fp in sorted(glob.glob(os.path.join(input_path, "**", "*.jsonl"), recursive=True)):
            yield from iter_json_lines(fp)
        for fp in sorted(glob.glob(os.path.join(input_path, "**", "*.json"), recursive=True)):
            yield from iter_json_lines(fp)


def write_docstore(out_path: str, rows: Iterable[Dict[str, Any]], limit: int = 0) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seen_hash: set[str] = set()
    n = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for r in rows:
            txt = (r.get("text") or "").strip()
            if not txt:
                continue
            h = content_hash(txt)
            if h in seen_hash:
                continue
            seen_hash.add(h)
            rec = {
                "id": h,
                "text": txt,
                "title": r.get("title") or None,
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if limit and n >= limit:
                break
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert MRQA dataset to docstore.jsonl (id,text,title)")
    ap.add_argument("--input", required=True, help="MRQA file or directory")
    ap.add_argument("--out", default="indexes/herb/docstore.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    total = write_docstore(args.out, iter_inputs(args.input), limit=args.limit)
    print(json.dumps({"written": total, "out": args.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()



