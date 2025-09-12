from __future__ import annotations
import os, json, argparse, hashlib
from pathlib import Path
from typing import Iterable, Dict, List


def content_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def iter_txt_files(store_dir: str, exts: List[str], include_all: bool) -> Iterable[Dict]:
    # 재귀적으로 순회하며 지정 확장자 수집(없으면 include_all 로 모든 파일 시도)
    exts_lc = set([e.lower().lstrip('.') for e in exts])
    for root, _, files in os.walk(store_dir):
        for name in sorted(files):
            if not include_all:
                if '.' in name:
                    if name.rsplit('.',1)[-1].lower() not in exts_lc:
                        continue
                else:
                    continue
            fp = os.path.join(root, name)
            try:
                text = Path(fp).read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not text:
                continue
            title = Path(fp).stem
            yield {"text": text, "title": title}


def write_docstore(out_path: str, rows: Iterable[Dict], limit: int = 0) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seen: set[str] = set()
    n = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for r in rows:
            txt = (r.get("text") or "").strip()
            if not txt:
                continue
            h = content_hash(txt)
            if h in seen:
                continue
            seen.add(h)
            rec = {"id": h, "text": txt, "title": r.get("title")}
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if limit and n >= limit:
                break
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert store/*.txt to docstore.jsonl (id,text,title)")
    ap.add_argument("--store", default="store", help="directory containing text files")
    ap.add_argument("--out", default="indexes/txt/docstore.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ext", action="append", default=["txt","TXT","md","text"], help="file extensions to include")
    ap.add_argument("--include-all", action="store_true", help="ingest all files regardless of extension")
    args = ap.parse_args()

    rows = list(iter_txt_files(args.store, args.ext, args.include_all))
    total = write_docstore(args.out, rows, args.limit)
    print(json.dumps({"written": total, "out": args.out, "scanned": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


