#!/usr/bin/env python3
"""
Freebase 트리플 데이터를 Neo4j에 직접 로드하는 스크립트
형식: subject_id \t predicate \t object_id
"""

from __future__ import annotations
import os, sys, argparse, json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from tqdm import tqdm

# Path hack
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)

from herb_rag_kit.graphdb.neo4j_store import Neo4jStore

def parse_freebase_line(line: str) -> Tuple[str, str, str] | None:
    """Parse a line in format: subject \t predicate \t object"""
    parts = line.strip().split('\t')
    if len(parts) != 3:
        return None
    s, p, o = parts
    return s.strip(), p.strip(), o.strip()

def clean_freebase_id(fb_id: str) -> str:
    """Convert /m/abc123 to m.abc123 for readability"""
    if fb_id.startswith('/m/'):
        return 'm.' + fb_id[3:]
    elif fb_id.startswith('/'):
        return fb_id[1:].replace('/', '.')
    return fb_id

def clean_predicate(pred: str) -> str:
    """Convert /domain/type/property to domain_type_property"""
    if pred.startswith('/'):
        pred = pred[1:]
    return pred.replace('/', '_').replace('.', '_')

def load_freebase_file(file_path: str, limit: int = 0) -> List[Dict]:
    """Load and parse Freebase triples from file"""
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if limit and i > limit:
                break
            
            parsed = parse_freebase_line(line)
            if not parsed:
                continue
                
            s, p, o = parsed
            triples.append({
                's': clean_freebase_id(s),
                'p': clean_predicate(p),
                'o': clean_freebase_id(o),
                's_type': 'ENTITY',
                'o_type': 'ENTITY',
                's_key': f"{clean_freebase_id(s)}|ENTITY",
                'o_key': f"{clean_freebase_id(o)}|ENTITY",
                'doc_id': f"freebase_{Path(file_path).stem}",
                'conf': 1.0
            })
    
    return triples

def batch_ingest_triples(store: Neo4jStore, triples: List[Dict], batch_size: int = 1000):
    """Batch insert triples to Neo4j"""
    
    # Ingest Cypher (same as lg_pipeline.py)
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
    
    total_inserted = 0
    for i in tqdm(range(0, len(triples), batch_size), desc="Ingesting batches"):
        batch = triples[i:i+batch_size]
        store.run_cypher(CYPHER_INGEST, {"rows": batch})
        total_inserted += len(batch)
    
    return total_inserted

def main():
    parser = argparse.ArgumentParser(description="Load Freebase triples into Neo4j")
    parser.add_argument("--input", required=True, help="Input Freebase file path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of lines to process (0=all)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for Neo4j insertion")
    parser.add_argument("--clear-db", action="store_true", help="Clear existing graph before loading")
    
    args = parser.parse_args()
    
    # Initialize Neo4j store
    store = Neo4jStore()
    store.create_schema()
    
    if args.clear_db:
        print("Clearing existing graph...")
        store.run_cypher("MATCH (n) DETACH DELETE n")
    
    # Load and parse triples
    print(f"Loading triples from {args.input}...")
    triples = load_freebase_file(args.input, args.limit)
    print(f"Parsed {len(triples)} triples")
    
    if not triples:
        print("No valid triples found!")
        return
    
    # Analyze data
    pred_counts = Counter(t['p'] for t in triples)
    print(f"Top 10 predicates: {pred_counts.most_common(10)}")
    
    # Insert into Neo4j
    print("Inserting triples into Neo4j...")
    total_inserted = batch_ingest_triples(store, triples, args.batch_size)
    
    # Get final stats
    sizes = store.graph_size()
    
    result = {
        "input_file": args.input,
        "lines_processed": len(triples),
        "triples_inserted": total_inserted,
        "final_graph_size": sizes,
        "top_predicates": dict(pred_counts.most_common(10))
    }
    
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
