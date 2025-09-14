# HERB RAG Kit — GraphRAG Quickstart (Gemini 2.5 Pro)

이 저장소는 텍스트 코퍼스에서 지식그래프를 구축하고(LLM 추출), Neo4j에 적재한 뒤 품질지표·시각화를 수행하는 최소 실행 파이프라인을 제공합니다.

## 1) 설치
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 폴더 구조(요약)
```
herb_rag_kit/
  scripts/
    convert_txt_to_docstore.py   # TXT→docstore.jsonl 변환
    lg_pipeline.py               # LLM 기반 추출/질의 파이프라인
    kg_quality_eval.py           # 그래프 품질 지표 산출
    neo4j_export_viz.py          # HTML 시각화
    neo4j_export_graph.py        # (신규) GEXF/GraphML/CSV/PNG 내보내기
    freebase_loader.py           # (신규) Freebase 트리플 직접 적재
  src/herb_rag_kit/
    graphdb/neo4j_store.py
    store/ (원본 데이터: test/train/valid 등)
indexes/
runs/
```

### Makefile 사용(선택)
```bash
make venv && make install
make neo4j-up
make docstore  # TXT만 포함하여 docstore 생성
make extract   # LLM 추출(환경변수에 GEMINI_API_KEY 필요)
make eval      # 품질지표
make viz       # HTML 시각화 저장
make export-graph FMT=gexf OUT=runs/kg.gexf  # GEXF/GraphML/CSV/PNG
make freebase-load FILE=herb_rag_kit/src/herb_rag_kit/store/train.txt LIMIT=0 CLEAR=1
```

## 2) 데이터 준비(TXT 5개 예시)
- 위치: `src/herb_rag_kit/store/{test.txt,text_cvsc.txt,text_emnlp.txt,train.txt,valid.txt}`
- 변환: TXT → `docstore.jsonl` (id,text,title)
```bash
python scripts/convert_txt_to_docstore.py \
  --store src/herb_rag_kit/store \
  --out indexes/txt/docstore.jsonl --include-all
```

## 3) Neo4j 실행(도커)
```bash
export NEO4J_PASSWORD='Neo4j-1717!'
docker start neo4j-herb || docker run -d --name neo4j-herb \
  -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH="neo4j/${NEO4J_PASSWORD}" neo4j:5
until (echo > /dev/tcp/127.0.0.1/7687) >/dev/null 2>&1; do sleep 1; done
```

## 4) 환경변수 설정(LLM/DB)
```bash
export GEMINI_API_KEY=AIzaSyCv2gHm_1veloZCVfs67kBrNlcVBx_zMLM
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD="$NEO4J_PASSWORD"
```

## 5) 추출 파이프라인 실행(정제·일관성·바이어스 완화 포함)
```bash
python scripts/lg_pipeline.py extract \
  --docstore indexes/txt/docstore.jsonl \
  --conf-threshold 0.6 --consistency-llm \
  --max-objects-per-sp 5 --max-per-entity 200
```

## 6) 품질지표 및 시각화
```bash
python scripts/kg_quality_eval.py
python scripts/neo4j_export_viz.py --out runs/kg_subgraph.html
```

## 7) FB15k‑237 전량 적재 및 활용(권장 경로)
- 데이터: 엔티티 14,541 · 관계 237 · 트리플 310,116 (train 272,115 / valid 17,535 / test 20,466)
- 이미 `herb_rag_kit/src/herb_rag_kit/store/{train,valid,test}.txt`에 포함되어 있으며, 다음으로 전량 적재/평가합니다.

```bash
source .venv/bin/activate
make neo4j-up

# 전량 적재(첫 파일만 CLEAR=1)
make freebase-load FILE=herb_rag_kit/src/herb_rag_kit/store/test.txt  LIMIT=0 CLEAR=1
make freebase-load FILE=herb_rag_kit/src/herb_rag_kit/store/train.txt LIMIT=0
make freebase-load FILE=herb_rag_kit/src/herb_rag_kit/store/valid.txt LIMIT=0

# 그래프 품질지표
make eval

# 시각화(HTML) 또는 내보내기(GEXF/GraphML/CSV/PNG)
make viz
make export-graph FMT=gexf OUT=runs/kg_1k.gexf \
  CY="MATCH (e1:Entity)-[r:RELATES]->(e2:Entity) RETURN e1,r,e2 LIMIT 1000"
```

## 8) KGE 베이스라인(선택: PyKEEN)
```bash
source .venv/bin/activate
pip install -U pykeen

# DistMult
python - << 'PY'
from pykeen.pipeline import pipeline
res = pipeline(dataset='FB15k237', model='DistMult', training_kwargs=dict(num_epochs=100))
res.save_to_directory('runs/kge/pykeen_distmult')
print(res.metric_results.to_str())
PY

# ComplEx
python - << 'PY'
from pykeen.pipeline import pipeline
res = pipeline(dataset='FB15k237', model='ComplEx', training_kwargs=dict(num_epochs=100))
res.save_to_directory('runs/kge/pykeen_complex')
print(res.metric_results.to_str())
PY
```

## 주의/팁
- LLM 추출 경로는 자연어 텍스트가 필요합니다. FB15k‑237은 구조화 삼중항이므로 LLM 없이 Cypher/Neo4j로 바로 활용하세요.
- LLM 질의 경로를 사용할 때는 `export GEMINI_API_KEY=...` 후 `scripts/lg_pipeline.py query --q "..."`를 실행하십시오.

## 트러블슈팅
- `extracted_total = 0`인 경우:
  - `--conf-threshold 0.5`로 완화, `LG_MAX_CHUNKS/ LG_CHUNK_CHARS` 조정
  - `GEMINI_API_KEY` 설정 확인, 네트워크 상태 확인
- Neo4j 연결 거부:
  - 컨테이너 기동/포트 대기 후 재시도(`7687`), 비밀번호 일치 확인

## 비고
- 키는 코드에 하드코딩하지 않고 환경변수로만 사용합니다.
