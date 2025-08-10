# compare — HERB RAG Methods (DRAGIN / DoTA-RAG / GraphRAG / HD-RAG)

기업형 RAG 벤치마크 **HERB (Salesforce/HERB)**를 기반으로, 서로 다른 RAG 방법론을 **동일 조건**에서 실험·비교하기 위한 미니 프레임워크입니다.  
포함된 방법론(라이트 구현):

- **DRAGIN-lite**: 동적 검색 트리거(RIND) + 질의 정식화(QFS)
- **DoTA-RAG-lite**: 쿼리 라우팅 → 하이브리드 검색 → 리랭킹
- **GraphRAG-lite**: 엔티티 그래프 기반 멀티-홉 수집
- **HD-RAG-lite**: 표(H-RCL 텍스트화) + 2단계 검색(LLM 재스코어)

공통 LLM: **Gemini 2.5 Pro** (`GEMINI_API_KEY` 필요)  
데이터: **HERB** (Hugging Face Hub에서 JSON 파일 직접 다운로드 → 로컬 파일 변환)

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

# 2) 패키지 경로 (src 패스 필요)
export PYTHONPATH=$PWD/src

# 3) 환경 변수(.env)
cp .env.example .env
# 편집기로 .env 열고 GEMINI_API_KEY=<당신의 키> 입력 (절대 커밋 금지)

# 4) HERB 데이터 인제스트(처음 1회)
mkdir -p ~/HERB_repo/data
python scripts/ingest_herb_hf.py --out_dir ~/HERB_repo/data
# -> ~/HERB_repo/data/{corpus/*.json, questions.jsonl, gold.jsonl}

# 5) 방법론 실행 (예: DoTA-RAG)
python scripts/run_dota.py --herb_root ~/HERB_repo --out runs/dota_r1.jsonl

# 6) 평가 (Fresh@K/TTI 포함)
python src/herb_rag_kit/eval/herb_eval_extras.py \
  --pred runs/dota_r1.jsonl \
  --gold ~/HERB_repo/data/gold.jsonl \
  --t0 2025-07-01T00:00:00Z \
  --k 1 5 10 \
  --out runs/results_dota_r1.json
