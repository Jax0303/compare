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

## 2) HERB 데이터 준비

```
git clone https://github.com/Jax0303/HERB.git
```

## 3) 인덱스 생성
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

각 모듈은 엄격한 타입힌트/예외 처리/로깅을 포함합니다.
