# HERB RAG Kit — GraphRAG Quickstart (Gemini 2.5 Pro)

이 저장소는 텍스트 코퍼스에서 지식그래프를 구축하고(LLM 추출), Neo4j에 적재한 뒤 품질지표·시각화를 수행하는 최소 실행 파이프라인을 제공합니다.

## 1) 설치
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
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
export GEMINI_API_KEY=YOUR_KEY
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

## 트러블슈팅
- `extracted_total = 0`인 경우:
  - `--conf-threshold 0.5`로 완화, `LG_MAX_CHUNKS/ LG_CHUNK_CHARS` 조정
  - `GEMINI_API_KEY` 설정 확인, 네트워크 상태 확인
- Neo4j 연결 거부:
  - 컨테이너 기동/포트 대기 후 재시도(`7687`), 비밀번호 일치 확인

## 비고
- 키는 코드에 하드코딩하지 않고 환경변수로만 사용합니다.
