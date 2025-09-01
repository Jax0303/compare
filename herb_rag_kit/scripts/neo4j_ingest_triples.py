import os, json, csv, argparse
from typing import Dict, Any, Iterable, List
from herb_rag_kit.graphdb.neo4j_store import Neo4jStore

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def iter_csv(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row

def norm_row(t: Dict[str, Any]) -> Dict[str, Any] | None:
    if not {"s","p","o"} <= set(t):
        return None
    s = str(t["s"]).strip(); p = str(t["p"]).strip(); o = str(t["o"]).strip()
    if not s or not p or not o:
        return None
    s_type = (str(t.get("s_type","THING")) or "THING").strip()
    o_type = (str(t.get("o_type","THING")) or "THING").strip()
    doc_id = t.get("doc_id")
    if doc_id is not None:
        doc_id = str(doc_id).strip() or None
    conf = t.get("conf", 1.0)
    try:
        conf = float(conf)
    except:
        conf = 1.0
    return {
        "s": s, "p": p, "o": o,
        "s_type": s_type, "o_type": o_type,
        "s_key": f"{s}|{s_type}", "o_key": f"{o}|{o_type}",
        "doc_id": doc_id, "conf": conf
    }

def iter_triples(path: str) -> Iterable[Dict[str, Any]]:
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        reader = iter_jsonl if ext == ".jsonl" else iter_csv if ext == ".csv" else None
        if reader is None:
            return
        for t in reader(path):
            nt = norm_row(t)
            if nt:
                yield nt
    else:
        for root, _, files in os.walk(path):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in (".jsonl", ".csv"):
                    continue
                yield from iter_triples(os.path.join(root, fn))

CYPHER = """
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL/CSV 파일 또는 디렉터리(재귀)")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true", help="DB에 쓰지 않고 카운트만")
    args = ap.parse_args()

    store = Neo4jStore()
    buf: List[Dict[str, Any]] = []
    total = 0
    seen = 0

    def flush():
        nonlocal buf, total
        if not buf:
            return
        if not args.dry_run:
            store.run_cypher(CYPHER, {"rows": buf})
        total += len(buf)
        buf = []
        print(f"[INGEST] total={total}")

    for t in iter_triples(args.input):
        seen += 1
        buf.append(t)
        if len(buf) >= args.batch:
            flush()
    flush()
    print(f"[OK] seen={seen}, ingested={total}{' (dry-run)' if args.dry_run else ''}")

if __name__ == "__main__":
    main()
