# LangChain 기반 RAG 서버 (고급)
문서 기반 QA와 명령 JSON 생성을 동시에 제공하는 FastAPI RAG 서버입니다. Parent-Document Retriever 구조, FlashRank Re-Ranking, guardrail/신뢰도 계산, intent 자동 분기, 명령 화이트리스트 검증을 포함합니다.

## 주요 기능
- 다중 포맷 문서 로딩 → 청킹 → 벡터 적재 (`ingest_langchain.py`)
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

## 폴더 구조
```
.
├── chains/               # RAG/command 체인
├── commands/             # 허용 명령 화이트리스트
├── docs/                 # 원본 문서
├── prompts/              # LLM 프롬프트
├── schemas/              # Pydantic 스키마
├── services/             # retrieval/rerank/parser/validator/confidence/intent 등
├── ingest_langchain.py   # 문서 적재 스크립트
├── rag_server.py         # FastAPI 엔트리 (/chat, /command, /ask)
├── query_test.py         # 검색만 단독 테스트
├── config.py             # 설정 (API 키는 환경 변수 권장)
└── chroma_db/            # 벡터 DB 저장 위치
```

## 설치
```bash
pip install -r requirements.txt
```

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
- 기본 청크: 800자, 오버랩 100자 (`config.py` / `ingest_langchain.py`)
- Parent/Child 저장: child→Chroma, parent→SQLite(`parent_docstore.sqlite`)
- 원본 문서 변경 시 재실행

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
1) ingest: 문서 로드 → `doc_id` 부여 → parent(2000/200)/child(800/100) 이중 청킹 → parent는 SQLite, child는 Chroma 저장  
2) 요청: child similarity 검색(initial_k=20) → FlashRank rerank → parent 복원·dedupe → parent별 최소 distance score → guardrail/신뢰도 계산  
3) `/chat`: parent 컨텍스트 포맷(문서별 900자, 전체 3,500자) → `_trim_context`(최대 12,000자) → LLM 응답 + 출처  
4) `/command`: 동일 컨텍스트 → LLM JSON → 파싱/화이트리스트 검증 → 신뢰도 낮으면 차단  
5) `/ask`: intent 분류 후 `/chat` 또는 `/command`

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
- 문제 발생 시 `docs/troubleshooting.txt` 참고

