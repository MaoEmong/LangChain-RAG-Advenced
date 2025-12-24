# LangChain 기반 RAG 시스템 (고급 버전)

LangChain과 Chroma를 활용해 **RAG QA**와 **명령 JSON 생성**을 모두 제공하는 FastAPI 서버 버전입니다. 검색 신뢰도 컷, confidence 계산, intent 분류, 명령 화이트리스트 검증 등을 포함합니다.

## 주요 기능
- 다중 포맷 문서 로딩(`.txt/.md/.pdf/.docx/.html`) 후 청킹 및 Chroma에 적재 (`ingest_langchain.py`)
- 2단계 검색 파이프라인: 벡터 검색 + FlashRank Re-Ranking으로 검색 정확도 향상
- Guardrail: top score 컷, 충분한 good hit 검사
- confidence 점수 계산 및 레벨(high/medium/low) 반환
- intent 분류(rule → LLM)로 explain/command 자동 라우팅 (`/ask`)
- 명령 JSON 생성 체인 + 화이트리스트 검증 (`/command`)
- 출처·score·미리보기 포함 RAG 답변 (`/chat`)

## 기술 스택
- Python 3, FastAPI, Uvicorn
- LangChain, LangChain OpenAI/Chroma/Text Splitters/Community
- Chroma (persisted vector DB)
- FlashRank (경량 Re-Ranking 모델)
- Pydantic

## 폴더 구조
```
.
├── chains/               # RAG/command 체인 정의
├── commands/             # 허용 명령 화이트리스트
├── docs/                 # 원본 문서
├── prompts/              # LLM 프롬프트 템플릿
├── schemas/              # Pydantic 스키마
├── services/             # vector DB, retrieval, rerank, parser, validator, confidence, intent
├── ingest_langchain.py   # 문서 적재 스크립트
├── rag_server.py         # FastAPI 진입점 (/chat, /command, /ask)
├── query_test.py         # 검색만 단독 테스트
├── config.py             # 설정 (API 키는 환경변수 사용 권장)
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

## 설치 & 환경 설정
1) 패키지 설치  
```bash
pip install -r requirements.txt
```

2) 환경 변수(.env 권장)  
```
OPENAI_API_KEY=your-api-key
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
CHROMA_DIR=./chroma_db
COLLECTION_NAME=my_rag_docs
TOP_K=4
# 선택: guard/confidence 파라미터는 config.py에서 조정
```

## 데이터 적재(ingest)
문서를 `docs/`에 넣은 뒤 실행:
```bash
python ingest_langchain.py
```
- 기본 청크: 600자, 오버랩 100자 (`ingest_langchain.py`에서 조정)
- 새 문서를 추가하거나 내용을 바꾼 경우에만 재실행하면 됩니다.

## 서버 실행
```bash
uvicorn rag_server:app --reload --port 8000
```

## API 엔드포인트
- `POST /chat` : RAG QA
  - 입력: `{"question": "..."}`  
  - 응답: `answer`, `sources`(source/score/preview), `guard`(reason, top_score, good_hits), `confidence`
- `POST /command` : 명령 JSON 제안
  - guardrail: 검색 결과/신뢰도 부족 시 차단, 화이트리스트 검증 실패 시 거부
  - 응답: `speech`, `actions[]`, `confidence`, `guard`(optional)
- `POST /ask` : intent 자동 분기
  - rule 기반 → LLM 보완으로 explain/command 결정 후 해당 로직 재사용

### 간단 호출 예시
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "RAG 파이프라인 설명해줘"}'

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "테크 스택 문서 열어줘"}'
```

## 동작 흐름
1) ingest: 문서 로드 → 청킹 → 임베딩 → Chroma 적재  
2) 요청 시: 벡터 검색(넓은 후보) → FlashRank Re-Ranking(정확도 향상) → guardrail 검사(top score, good hits) → confidence 계산  
3) `/chat`: Re-Ranking된 문서로 컨텍스트 포맷팅 → LLM 답변 + 출처 반환  
4) `/command`: Re-Ranking된 문서로 컨텍스트 구성 → LLM JSON 생성 → 파싱/검증 → 허용 명령만 반환  
5) `/ask`: intent 분류 후 `/chat` 또는 `/command` 실행

## 주요 설정 포인트
- `ingest_langchain.py`: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `COLLECTION_NAME`
- `config.py`: `TOP_K`, `TOP_SCORE_MAX`, `MIN_GOOD_HITS`, `GOOD_HIT_SCORE_MAX`, `CONF_SCORE_MIN/MAX`
- `rag_server.py`: `INITIAL_K` (Re-Ranking을 위한 초기 후보 개수, 기본값: 20)
- `commands/registry.py`: 허용 명령 및 필수 args 정의

## 보안/운영 팁
- API 키는 환경변수나 비밀 관리 서비스로 주입
- 벡터 DB(`chroma_db/`)는 필요 시 백업/재생성 가능
- 문제 발생 시 `docs/troubleshooting.txt` 참고

