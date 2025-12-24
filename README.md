# LangChain 기반 RAG 시스템 (고급 버전)

LangChain과 Chroma를 활용해 **RAG QA**와 **명령 JSON 생성**을 모두 제공하는 FastAPI 서버 버전입니다. 검색 신뢰도 컷, confidence 계산, intent 분류, 명령 화이트리스트 검증 등을 포함합니다.

## 주요 기능
- 다중 포맷 문서 로딩(`.txt/.md/.pdf/.docx/.html`) 후 청킹 및 Chroma에 적재 (`ingest_langchain.py`)
- MMR 기반 retriever + guardrail: top score 컷, 충분한 good hit 검사
- confidence 점수 계산 및 레벨(high/medium/low) 반환
- intent 분류(rule → LLM)로 explain/command 자동 라우팅 (`/ask`)
- 명령 JSON 생성 체인 + 화이트리스트 검증 (`/command`)
- 출처·score·미리보기 포함 RAG 답변 (`/chat`)

## 기술 스택
- Python 3, FastAPI, Uvicorn
- LangChain, LangChain OpenAI/Chroma/Text Splitters/Community
- Chroma (persisted vector DB)
- Pydantic

## 폴더 구조
```
.
├── chains/               # RAG/command 체인 정의
├── commands/             # 허용 명령 화이트리스트
├── docs/                 # 원본 문서
├── prompts/              # LLM 프롬프트 템플릿
├── schemas/              # Pydantic 스키마
├── services/             # vector DB, parser, validator, confidence, intent
├── ingest_langchain.py   # 문서 적재 스크립트
├── rag_server.py         # FastAPI 진입점 (/chat, /command, /ask)
├── query_test.py         # 검색만 단독 테스트
├── config.py             # 설정 (API 키는 환경변수 사용 권장)
└── chroma_db/            # 벡터 DB 저장 위치
```

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
`config.py`에 키를 하드코딩하지 말고 환경변수/비밀 관리 서비스를 사용하세요. 버전 관리 시 `config.py`를 커밋에서 제외하세요.

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
2) 요청 시: MMR 검색 → guardrail 검사(top score, good hits) → confidence 계산  
3) `/chat`: 컨텍스트 포맷팅 → LLM 답변 + 출처 반환  
4) `/command`: 컨텍스트 → LLM JSON 생성 → 파싱/검증 → 허용 명령만 반환  
5) `/ask`: intent 분류 후 `/chat` 또는 `/command` 실행

## 주요 설정 포인트
- `ingest_langchain.py`: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `COLLECTION_NAME`
- `config.py`: `TOP_K`, `TOP_SCORE_MAX`, `MIN_GOOD_HITS`, `GOOD_HIT_SCORE_MAX`, `CONF_SCORE_MIN/MAX`
- `commands/registry.py`: 허용 명령 및 필수 args 정의

## 보안/운영 팁
- API 키는 환경변수나 비밀 관리 서비스로 주입
- 벡터 DB(`chroma_db/`)는 필요 시 백업/재생성 가능
- 문제 발생 시 `docs/troubleshooting.txt` 참고

