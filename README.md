# HERB RAG Kit — DRAGIN / DoTA‑RAG / GraphRAG / HD‑RAG (Gemini 2.5 Pro)

**목표**: HERB(Heterogeneous Enterprise RAG Benchmark)로 *새 데이터 처리력*을 공정하게 비교.
- 공통 LLM: **Gemini 2.5 Pro**
- 공통 지표: EM, F1, Hit@K, Fresh@K(T0 이후), 지연(평균/중앙/95p), (옵션) Correctness/Faithfulness

## 1) 설치
```bash
python3 -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# .env 파일에서 GEMINI_API_KEY=... 값을 채워주세요.
```

> 당신이 준 키는 코드에 절대 하드코딩하지 않습니다. 반드시 환경변수(`.env` 또는 시스템 환경)로만 사용합니다.

## 2) Quickstart — TXT 코퍼스 → GraphRAG 파이프라인

```bash
# 0) 가상환경
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) TXT → docstore (id,text,title)
python herb_rag_kit/scripts/convert_txt_to_docstore.py \
  --store herb_rag_kit/src/herb_rag_kit/store \
  --out indexes/txt/docstore.jsonl --include-all

# 2) Neo4j 실행(로컬 도커)
export NEO4J_PASSWORD='Neo4j-1717!'
docker start neo4j-herb || docker run -d --name neo4j-herb \
  -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH="neo4j/${NEO4J_PASSWORD}" neo4j:5
until (echo > /dev/tcp/127.0.0.1/7687) >/dev/null 2>&1; do sleep 1; done
docker exec -it neo4j-herb cypher-shell -u neo4j -p "$NEO4J_PASSWORD" 'RETURN 1;'

# 3) 환경변수(LLM/Neo4j)
export GEMINI_API_KEY=YOUR_KEY
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD="$NEO4J_PASSWORD"

# 4) 추출→정제→일관성→바이어스완화→Neo4j 적재
python herb_rag_kit/scripts/lg_pipeline.py extract \
  --docstore indexes/txt/docstore.jsonl \
  --conf-threshold 0.6 --consistency-llm \
  --max-objects-per-sp 5 --max-per-entity 200

# 5) 품질지표
python herb_rag_kit/scripts/kg_quality_eval.py

# 6) 서브그래프 HTML 시각화
python herb_rag_kit/scripts/neo4j_export_viz.py \
  --cypher "MATCH (e1:Entity)-[r:RELATES]->(e2:Entity) RETURN e1,r,e2 LIMIT 300" \
  --out runs/kg_subgraph.html
```

## 3) (선택) HERB 데이터 준비/인덱스
```bash
python scripts/index_corpus.py --herb_root /path/to/HERB --out_dir .cache/index
```

## 4) 방법론 실행 예
- DRAGIN (동적 검색 트리거)
```bash
python scripts/run_dragin.py --herb_root /path/to/HERB --index .cache/index --out runs/dragin_r1.jsonl
```
- DoTA‑RAG (쿼리 재작성→동적 라우팅→하이브리드 검색)
```bash
python scripts/run_dota.py --herb_root /path/to/HERB --index .cache/index --out runs/dota_r1.jsonl
```
- GraphRAG (그래프 인덱싱 + 멀티‑홉 검색)
```bash
python scripts/run_graphrag.py --herb_root /path/to/HERB --index .cache/index --out runs/graphrag_r1.jsonl
```
- HD‑RAG (H‑RCL 표요약 + 2단계 검색 + RECAP)
```bash
python scripts/run_hdrag.py --herb_root /path/to/HERB --index .cache/index --out runs/hdrag_r1.jsonl
```

## 5) 평가(Fresh@K/TTI 포함)
```bash
python src/herb_rag_kit/eval/herb_eval_extras.py   --pred runs/dota_r1.jsonl runs/dota_r2.jsonl   --gold /path/to/HERB/data/gold.jsonl   --t0 2025-07-01T00:00:00Z   --k 1 5 10   --tti_metric hit@1 --tti_threshold 0.5   --out results_dota_tti.json

**목표**: HERB(Heterogeneous Enterprise RAG Benchmark)로 *새 데이터 처리력*을 공정하게 비교.
- 공통 LLM: **Gemini 2.5 Pro**
- 공통 지표: EM, F1, Hit@K, Fresh@K(T0 이후), 지연(평균/중앙/95p), (옵션) Correctness/Faithfulness
```

## 구현 메모
- **DRAGIN-lite**: RIND=자기일관성 기반 불확실성(다중 샘플 다양도) + 키워드 힌트, QFS=키프레이즈/명사구 추출(+LLM 리파인)으로 쿼리 생성.
- **DoTA-lite**: 라우터=메타/토픽 키워드 + (옵션) LLM 클래시파이어, Stage3=BM25+임베딩 하이브리드 검색 후 rerank.
- **GraphRAG-lite**: spaCy NER → entity graph(networkx), 질문 엔티티 확장 멀티‑홉 서브그래프 수집.
- **HD‑RAG-lite**: H‑RCL 표요약(행/열/경로) 텍스트화 → 앙상블 검색(BM25+임베딩) → LLM 기반 re‑score.

## Graph Visualization (PyVis, community meta-graph)

This repo includes inline visualization utilities and Neo4j subgraph HTML export.

### Requirements
```bash
pip install -U pyvis jinja2 python-louvain scikit-learn

Generate visualizations
python scripts/viz_inline.py


Artifacts:

runs/graphrag_viz_inline.html — filtered main graph (backbone + k-core + ForceAtlas2)

runs/community_graph.html — community meta-graph (cluster-level map)

runs/graphrag_filtered.gexf — filtered graph for Gephi

Open in VS Code via Live Preview / Live Server or:

python -m http.server -d runs 8899
# then open http://localhost:8899/graphrag_viz_inline.html

Tuning knobs (inside scripts/viz_inline.py)

TOPK_HUBS (default 100): number of hub anchors (higher → larger graph)

RADIUS (default 1): r-hop neighbors from hubs (2 increases density a lot)

BACKBONE_TOPK (default 3): keep top-k weighted edges per node

KCORE_K (default 2): k-core filter (0 to skip)

TARGET_MAX_NODES (default 700): soft cap with auto degree trim

LABEL_MIN_DEG, LABEL_TOP: label density control

Meta-graph cleanup: COMM_EDGE_W_MIN, TOPM_COMM_EDGES

Notes

Outputs in runs/ are ignored by git via .gitignore. For Neo4j subgraph export:
```bash
python herb_rag_kit/scripts/neo4j_export_viz.py --out runs/kg_subgraph.html
```

To re-generate from a different GEXF, change GEXF_IN in scripts/viz_inline.py.