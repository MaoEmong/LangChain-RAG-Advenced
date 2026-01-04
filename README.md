# LangChain 기반 RAG 서버 (고급)

문서 기반 QA와 명령 JSON 생성을 동시에 제공하는 FastAPI RAG 서버입니다. Parent-Document Retriever 구조, FlashRank Re-Ranking, guardrail/신뢰도 계산, intent 자동 분기, 명령 화이트리스트 검증을 포함합니다.

## 주요 기능

- **다중 포맷 문서 로딩** → 청킹 → 벡터 적재 (`ingest_langchain.py`)
  - PDF (텍스트 PDF / 스캔 PDF 자동 구분)
  - 텍스트 파일 (TXT, MD)
  - Word 문서 (DOCX)
  - HTML 파일
- **PDF 자동 처리**:
  - 텍스트 PDF: PyMuPDF로 직접 텍스트 추출
  - 스캔 PDF: Tesseract OCR을 사용한 이미지 텍스트 인식 (한국어/영어 지원)
- Parent-Document Retriever: child는 검색/랭킹, parent는 LLM 컨텍스트 (Chroma + SQLite)
- 2단계 검색: 벡터 검색 → FlashRank Re-Ranking
- Guardrail & Confidence: top score 컷, good hit 개수, 신뢰도 점수/레벨 반환
- Intent 분류(`/ask`): explain/command로 자동 라우팅
- 명령 JSON 생성 + 파싱 + 화이트리스트 검증(`/command`)
- 출처/score/미리보기 포함 RAG 응답(`/chat`)

## 기술 스택

- Python 3, FastAPI, Uvicorn
- LangChain(OpenAI, Chroma, Text Splitters, Community)
- Parent-Document Retriever 패턴 (Chroma child + SQLite parent)
- FlashRank Re-Ranker
- Pydantic
- PyMuPDF (PDF 텍스트 추출)
- Tesseract OCR (스캔 PDF 이미지 인식)

## 폴더 구조

```
.
├── chains/               # RAG/command 체인
├── commands/             # 허용 명령 화이트리스트
├── docs/                 # 원본 문서
├── loaders/              # 문서 로더 (PDF, TXT, MD 등)
│   ├── auto_loader.py    # 자동 문서 로더 (PDF 자동 구분)
│   ├── pdf_detector.py   # PDF 종류 판별기 (텍스트/스캔)
│   ├── pdf_text_loader.py # 텍스트 PDF 로더
│   ├── pdf_scan_loader.py # 스캔 PDF 로더 (OCR)
│   └── rules.py          # 파일 확장자별 로더 규칙
├── ocr/                  # OCR 엔진
│   ├── base.py           # OCR 베이스 클래스
│   ├── tesseract_ocr.py  # Tesseract OCR 구현
│   └── dummy_ocr.py      # 더미 OCR (테스트용)
├── preprocess/           # 전처리 유틸리티
│   └── text_cleaner.py   # 텍스트 정리 (공백, 개행 등)
├── prompts/              # LLM 프롬프트
├── schemas/              # Pydantic 스키마
├── services/             # retrieval/rerank/parser/validator/confidence/intent 등
├── ingest_langchain.py   # 문서 적재 스크립트
├── rag_server.py         # FastAPI 엔트리 (/chat, /command, /ask)
├── query_test.py         # 검색만 단독 테스트
├── test_loaders.py       # 문서 로더 테스트
├── test_retrieval.py     # 검색 기능 테스트
├── debug_chroma_scan.py  # ChromaDB 스캔 문서 디버깅
├── config.py             # 설정 (API 키는 환경 변수 권장)
└── chroma_db/            # 벡터 DB 저장 위치
```

## 주요 파일 설명

### 루트 디렉토리

- **`rag_server.py`**: FastAPI 서버 메인 파일
  - `/chat`, `/command`, `/ask` 엔드포인트 제공
  - 2단계 검색 파이프라인 (벡터 검색 + FlashRank Re-Ranking) 실행
  - Guardrail 및 Confidence 계산
- **`ingest_langchain.py`**: 문서 수집 및 벡터DB 구축 스크립트
  - `docs/` 폴더의 문서를 읽어서 ChromaDB에 저장
  - 문서 청킹 및 임베딩 변환
  - 문서가 변경되었을 때만 실행하면 됨
- **`query_test.py`**: 벡터 검색 기능만 단독으로 테스트하는 스크립트
  - 서버 실행 없이 검색 결과 확인 가능
- **`config.py`**: 프로젝트 전역 설정 파일
  - OpenAI API 키, 모델 설정, Guardrail 파라미터 등

### loaders/

- **`auto_loader.py`**: 자동 문서 로더
  - 폴더 내 모든 문서를 자동으로 감지하고 적절한 로더 선택
  - PDF 파일의 경우 텍스트 PDF와 스캔 PDF를 자동 구분하여 처리
  - 문서 메타데이터 자동 추가 (doc_id, source, kind, domain 등)
- **`pdf_detector.py`**: PDF 종류 판별기
  - 텍스트 레이어 존재 여부를 확인하여 텍스트 PDF/스캔 PDF 자동 판별
- **`pdf_text_loader.py`**: 텍스트 PDF 로더
  - PyMuPDF를 사용하여 텍스트 레이어에서 직접 텍스트 추출
  - 파일당 Document 1개로 통합 로드
- **`pdf_scan_loader.py`**: 스캔 PDF 로더
  - Tesseract OCR을 사용하여 이미지에서 텍스트 추출
  - 한국어/영어 다국어 지원
  - 파일당 Document 1개로 통합 로드
- **`rules.py`**: 파일 확장자별 로더 규칙
  - TXT, MD, PDF, DOCX, HTML 등의 로더 매핑

### ocr/

- **`base.py`**: OCR 베이스 클래스
  - OCR 엔진의 공통 인터페이스 정의
- **`tesseract_ocr.py`**: Tesseract OCR 구현
  - PDF를 이미지로 렌더링한 후 OCR 수행
  - 다국어 지원 (한국어, 영어 등)
- **`dummy_ocr.py`**: 더미 OCR (테스트용)

### preprocess/

- **`text_cleaner.py`**: 텍스트 정리 유틸리티
  - 개행 문자 통일, 과도한 공백 제거 등

### chains/

- **`rag_chain.py`**: RAG 체인 구성 모듈
  - 검색된 문서를 컨텍스트로 변환하는 함수 제공
  - LangChain 체인으로 질문 → 검색 → 답변 생성 파이프라인 구성
- **`command_chain.py`**: 명령 생성 체인 구성 모듈
  - 자연어 명령을 JSON 형식으로 변환하는 체인 구성

### services/

- **`vector_store.py`**: 벡터 저장소 생성 및 관리
  - ChromaDB 벡터 저장소 초기화 및 로드
- **`retrieval.py`**: 통합 검색 및 Re-Ranking 모듈
  - 2단계 검색 파이프라인: 벡터 검색 → FlashRank Re-Ranking
  - 원본 distance score를 rerank된 문서에 매핑
- **`rerank_flashrank.py`**: FlashRank Re-Ranker 래퍼
  - FlashRank를 사용하여 검색 결과 재정렬
  - 경량 모델로 빠른 처리 속도 제공
- **`confidence.py`**: 검색 결과 신뢰도 계산
  - 최상위 문서 점수와 좋은 문서 개수를 종합하여 신뢰도 계산
- **`intent_classifier.py`**: 사용자 의도 분류기
  - Rule 기반 + LLM 기반 하이브리드 방식
  - "command" 또는 "explain"으로 분류
- **`command_parser.py`**: 명령 JSON 파서
  - LLM이 생성한 JSON 문자열을 파싱 및 검증
- **`command_validator.py`**: 명령 검증기
  - 화이트리스트 기반으로 허용된 명령만 실행 가능하도록 검증

### commands/

- **`registry.py`**: 명령 화이트리스트 레지스트리
  - 서버가 실행을 허용하는 명령 목록 정의
  - 각 명령의 필수 인자 목록 포함

### schemas/

- **`command.py`**: 명령 스키마 정의
  - CommandResponse, CommandAction Pydantic 모델
- **`intent.py`**: 의도 분류 결과 스키마 정의
  - IntentResult Pydantic 모델

### prompts/

- **`command_prompt.py`**: 명령 생성 프롬프트 템플릿
  - 자연어 명령을 JSON으로 변환하기 위한 프롬프트
- **`intent_prompt.py`**: 의도 분류 프롬프트 템플릿
  - 사용자 입력을 "command" 또는 "explain"으로 분류하기 위한 프롬프트

### 테스트 파일

- **`test_loaders.py`**: 문서 로더 테스트 스크립트
  - PDF 종류 판별 테스트
  - 텍스트 PDF 추출 샘플 출력
  - 스캔 PDF OCR 샘플 출력
  - AutoLoader 전체 요약
- **`test_retrieval.py`**: 검색 기능 테스트 스크립트
  - ChromaDB에서 Child 문서 검색
  - doc_id별 랭킹
  - SQLiteDocStore에서 Parent 문서 복원
- **`query_test.py`**: ParentDocumentRetriever 동작 테스트
  - 대화형 질의응답 테스트
- **`debug_chroma_scan.py`**: ChromaDB 스캔 문서 디버깅 스크립트

## 설치 & 환경 설정

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. Tesseract OCR 설치 (스캔 PDF 처리용)

스캔 PDF 파일을 처리하려면 Tesseract OCR이 필요합니다.

**Windows:**

1. [Tesseract OCR 설치 프로그램](https://github.com/UB-Mannheim/tesseract/wiki) 다운로드 및 설치
2. 기본 설치 경로: `C:\Program Files\Tesseract-OCR\tesseract.exe`
3. `loaders/auto_loader.py`와 `ocr/tesseract_ocr.py`에서 경로 확인/수정

**Linux (Ubuntu/Debian):**

```bash
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-kor  # 한국어 지원
```

**macOS:**

```bash
brew install tesseract
brew install tesseract-lang  # 다국어 지원
```

**참고:**

- 한국어 지원을 위해 한국어 언어팩 설치 권장
- 설치 후 언어 코드: `"eng+kor"` (영어+한국어)

## 환경 변수(.env 권장)

```
OPENAI_API_KEY=your-api-key
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
CHROMA_DIR=./chroma_db
COLLECTION_NAME=my_rag_docs
TOP_K=4
# guard/confidence 파라미터는 config.py 참고
```

## 데이터 적재(ingest)

문서를 `docs/`에 넣고 실행:

```bash
python ingest_langchain.py
```

### 지원 파일 형식

- **PDF**: 텍스트 PDF와 스캔 PDF 자동 구분
  - 텍스트 PDF: PyMuPDF로 직접 텍스트 추출
  - 스캔 PDF: Tesseract OCR로 이미지 텍스트 인식
- **텍스트 파일**: TXT, MD
- **Word 문서**: DOCX
- **HTML 파일**: HTML, HTM

### 처리 방식

- 기본 청크: 800자, 오버랩 100자 (`config.py` / `ingest_langchain.py`)
- Parent/Child 저장: child→Chroma, parent→SQLite(`parent_docstore.sqlite`)
- PDF 파일: 파일당 Parent 문서 1개로 통합 (페이지별 분할 없음)
- 메타데이터: 각 문서에 `doc_id`, `source`, `kind`, `domain` 등 자동 추가
  - `kind`: 문서 종류 (pdf_text, pdf_scan_ocr, txt, md 등)
  - `domain`: 도메인 태그 (pdf_text, ocr_scan 등)
- 원본 문서 변경 시 재실행

### 테스트

문서 로더 기능 테스트:

```bash
python test_loaders.py
```

## 서버 실행

```bash
uvicorn rag_server:app --reload --port 8000
```

## API

- `POST /chat` : RAG QA  
  입력 `{"question":"..."}` → `answer`, `sources`(source/score/preview), `guard`(reason/top_score/good_hits), `confidence`
- `POST /command` : 명령 JSON 제안  
  검색/신뢰도 부족 시 차단, 화이트리스트 검증 실패 시 거부 → `speech`, `actions[]`, `confidence`, `guard`
- `POST /ask` : intent 자동 분기  
  rule+LLM로 explain/command 결정 후 해당 로직 수행

### 호출 예시

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "RAG 파이프라인 설명"}'

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "테크 스택 문서 열어줘"}'
```

## 동작 흐름

### 1. 문서 수집 (Ingest)

1. `docs/` 폴더에서 문서 자동 탐색
2. PDF 파일: 텍스트 PDF/스캔 PDF 자동 판별
   - 텍스트 PDF: PyMuPDF로 텍스트 추출
   - 스캔 PDF: Tesseract OCR로 이미지 텍스트 인식
3. 각 파일에 `doc_id` 부여 (파일별 고유 ID)
4. 파일당 Parent 문서 1개 생성 (모든 페이지 통합)
5. Parent 문서를 Child 청크로 분할 (800자, 오버랩 100자)
6. Parent는 SQLite 저장, Child는 ChromaDB에 벡터로 저장

### 2. 검색 요청

1. Child 문서 similarity 검색 (initial_k=20)
   - OCR 키워드 질의 시 OCR 문서 우선 검색 (키워드 바이어스)
2. FlashRank Re-ranking으로 결과 재정렬
3. Parent 문서 복원 및 중복 제거
4. Parent별 최소 distance score 계산
5. Guardrail 및 신뢰도 계산

### 3. `/chat` 엔드포인트

1. Parent 컨텍스트 포맷 (문서별 900자, 전체 3,500자)
2. `_trim_context`로 최대 12,000자 제한
3. LLM 응답 생성 + 출처 정보 반환

### 4. `/command` 엔드포인트

1. 동일 컨텍스트 사용
2. LLM이 JSON 형식 명령 생성
3. JSON 파싱 및 화이트리스트 검증
4. 신뢰도 낮으면 차단

### 5. `/ask` 엔드포인트

1. Intent 분류 (Rule 기반 + LLM 기반 하이브리드)
2. "command" 또는 "explain"으로 분류
3. 분류 결과에 따라 `/chat` 또는 `/command` 실행

## 주요 설정 포인트

- `ingest_langchain.py`: Parent/Child split(`build_parent_retriever`), `COLLECTION_NAME`
- `config.py`: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`, `TOP_K`, `TOP_SCORE_MAX`, `MIN_GOOD_HITS`, `GOOD_HIT_SCORE_MAX`, `CONF_SCORE_MIN/MAX`, `DOCSTORE_PATH`
- `rag_server.py`: `INITIAL_K=20`, `_trim_context` 최대 12,000자
- `services/retrieval.py`: `fetch_multiplier=3`로 rerank 범위를 top_k보다 넓혀 parent dedupe 손실 완화
- `commands/registry.py`: 허용 명령/필수 args 정의

## 아키텍처 메모 (Parent-Document Retriever)

- 검색은 child 청크(Chroma), LLM 컨텍스트는 parent(SQLite)
- 흐름: child 검색 → FlashRank rerank → parent 복원/중복 제거 → parent별 최소 distance score → guard/confidence → 컨텍스트 컷 → LLM

## 운영/보안 팁

- API 키는 환경 변수/비밀 관리 서비스 사용
- `chroma_db/`는 필요 시 재생성 가능(ingest 재실행)
- PDF 파일 처리:
  - 텍스트 PDF는 빠르게 처리 가능
  - 스캔 PDF는 OCR 처리로 시간이 오래 걸릴 수 있음 (파일 크기 및 페이지 수에 비례)
  - OCR 정확도를 높이려면 고해상도 PDF 사용 권장
- 문제 발생 시 `docs/troubleshooting.txt` 참고

## 주요 특징

### PDF 자동 구분

- 텍스트 레이어 존재 여부를 자동으로 판별
- 텍스트 PDF: 빠른 직접 추출
- 스캔 PDF: OCR을 통한 이미지 텍스트 인식

### OCR 키워드 검색 최적화

- 운송장 번호, 트래킹 코드 등의 키워드 검색 시 OCR 문서 우선 검색
- 키워드 바이어스를 통한 검색 정확도 향상

### Parent-Child 구조

- 검색은 작은 Child 청크로 정확도 향상
- LLM 컨텍스트는 전체 Parent 문서로 제공하여 컨텍스트 유지
