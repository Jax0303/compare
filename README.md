# compare — HERB RAG Methods (DRAGIN / DoTA-RAG / GraphRAG / HD-RAG)

기업형 RAG 벤치마크 **HERB (Salesforce/HERB)**를 기반으로, 서로 다른 RAG 방법론을 **동일 조건**에서 실험·비교하기 위한 미니 프레임워크입니다.  
포함된 방법론(라이트 구현):

- **DRAGIN-lite**: 동적 검색 트리거(RIND) + 질의 정식화(QFS)
- **DoTA-RAG-lite**: 쿼리 라우팅 → 하이브리드 검색 → 리랭킹
- **GraphRAG-lite**: 엔티티 그래프 기반 멀티-홉 수집
- **HD-RAG-lite**: 표(H-RCL 텍스트화) + 2단계 검색(LLM 재스코어)

공통 LLM: **Gemini 2.5 Pro** (`GEMINI_API_KEY` 필요)
데이터: **HERB**

---

## Directory
- `herb_rag_kit/`  
  - `scripts/` : 실행 스크립트 (`run_*.py`, `ingest_herb_hf.py`)  
  - `src/herb_rag_kit/` : 라이브러리 코드(메서드/검색/LLM 클라이언트/평가)  
  - `runs/` : 실행 결과(JSONL)  
  - `requirements.txt`, `.env.example`, `README.md`  

---

## Prerequisites
- Python 3.11+ (권장 3.12)
- Linux/macOS/WSL
- Google Generative AI 키: `GEMINI_API_KEY`
- 인터넷(처음 1회: HERB 파일 다운로드)

---

## Quickstart

```bash
# 0) 프로젝트 루트로 이동
cd ~/compare/herb_rag_kit

# 1) 가상환경 + 설치
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt datasets

# 2) (선택) 패키지 경로
# `run_*.py` 스크립트는 자동으로 `src` 경로를 추가하므로 일반 실행에는 필요 없습니다.
# 평가 스크립트 등을 직접 실행할 때만 설정하면 됩니다.
export PYTHONPATH=$PWD/src

# 3) 환경 변수(.env)
cp .env.example .env
# 편집기로 .env 열고 GEMINI_API_KEY=<당신의 키> 입력 (절대 커밋 금지)

# 4) HERB 데이터셋 가져오기 (GitHub)
git clone https://github.com/Jax0303/HERB.git ~/HERB_repo

# 5) 인덱싱/임베딩 + 파이프라인 실행 (예: DoTA-RAG, 10개 질문만)
python scripts/run_dota.py --herb_root ~/HERB_repo --out runs/dota_dbg.jsonl --limit 10
# 최초 실행 시 corpus 전체에 대한 BM25/임베딩 인덱스를 생성하고 `.cache/`에 저장합니다.
# 이후 실행부터는 캐시가 재사용되며 새 문서만 추가 임베딩합니다.

# 6) 다른 방법론 실행 (예: GraphRAG, 5개 질문)
python scripts/run_graphrag.py --herb_root ~/HERB_repo --out runs/graphrag_dbg.jsonl --limit 5

# 7) 평가 (Fresh@K/TTI 포함)
python src/herb_rag_kit/eval/herb_eval_extras.py \
  --pred runs/dota_dbg.jsonl \
  --gold ~/HERB_repo/data/gold.jsonl \
  --t0 2025-07-01T00:00:00Z \
  --k 1 5 10 \
  --out runs/results_dota_dbg.json
