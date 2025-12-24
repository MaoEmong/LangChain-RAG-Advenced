# LangChain 기반 RAG 시스템

LangChain과 ChromaDB를 활용한 기본적인 Retrieval-Augmented Generation (RAG) 시스템입니다.

## 📋 프로젝트 개요

이 프로젝트는 RAG 시스템의 핵심 구조를 이해하고 구현하기 위한 학습용 프로젝트입니다. 문서를 벡터로 변환하여 저장하고, 사용자의 질문과 관련된 문서를 검색한 뒤 OpenAI 모델을 통해 답변을 생성합니다.

### RAG 프로세스

1. **문서 수집 및 전처리**: `docs` 폴더의 문서들을 로드
2. **문서 임베딩 및 벡터 DB 저장**: 텍스트를 청킹하고 벡터로 변환하여 ChromaDB에 저장
3. **질문 임베딩**: 사용자 질문을 벡터로 변환
4. **유사 문서 검색**: 벡터 유사도 기반으로 관련 문서 검색
5. **검색 결과 기반 답변 생성**: 검색된 문서를 컨텍스트로 사용하여 답변 생성

## 🛠️ 기술 스택

### Backend
- Python 3
- LangChain
- Chroma Vector Database

### AI / ML
- OpenAI API (임베딩 및 채팅 모델)
- LangChain 텍스트 스플리터 (문서 청킹)
- Embedding 기반 유사도 검색

## 📁 프로젝트 구조

```
MyPython/
├── docs/                    # 문서 저장 폴더
│   ├── intro.txt
│   ├── project_overview.txt
│   ├── rag_concept.txt
│   ├── tech_stack.md
│   └── ...
├── chroma_db/               # ChromaDB 벡터 데이터베이스 저장 경로
├── config.py                # 설정 파일 (API 키 등)
├── ingest_langchain.py      # 문서 수집 및 벡터DB 구축 스크립트
├── query_test.py            # 벡터 검색 테스트 스크립트
└── README.md
```

## 🚀 시작하기

### 1. 환경 설정

필요한 패키지를 설치합니다:

```bash
pip install langchain langchain-openai langchain-chroma langchain-community
```

### 2. 설정 파일 구성

⚠️ **중요**: `config.py`에는 개인 API 키가 포함되어 있습니다. 보안을 위해 다음 중 하나를 권장합니다:

#### 방법 1: 환경 변수 사용 (권장)

`config.py`를 다음과 같이 수정:

```python
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "my_rag_docs"
TOP_K = 4
```

환경 변수 설정:
```bash
# Windows
set OPENAI_API_KEY=your-api-key-here

# Linux/Mac
export OPENAI_API_KEY=your-api-key-here
```

#### 방법 2: .env 파일 사용

`.env` 파일 생성:
```
OPENAI_API_KEY=your-api-key-here
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
```

`python-dotenv` 설치:
```bash
pip install python-dotenv
```

`config.py`에서 로드:
```python
from dotenv import load_dotenv
load_dotenv()
```

### 3. 문서 준비

`docs` 폴더에 처리할 문서 파일들을 넣습니다. 지원되는 파일 형식:
- `.txt` (텍스트 파일)
- `.md` (마크다운)
- `.pdf` (PDF 문서)
- `.docx` (Word 문서)
- `.html` / `.htm` (HTML 파일)

### 4. 벡터 데이터베이스 구축

문서를 로드하고 벡터DB에 저장합니다:

```bash
python ingest_langchain.py
```

이 스크립트는:
- `docs` 폴더의 모든 문서를 읽습니다
- 문서를 작은 청크(chunk)로 분할합니다 (기본: 1500자, 오버랩: 150자)
- 각 청크를 임베딩 벡터로 변환합니다
- ChromaDB에 저장합니다

**참고**: 문서가 변경되었을 때만 다시 실행하면 됩니다.

### 5. 검색 테스트

벡터 검색 기능을 테스트합니다:

```bash
python query_test.py
```

질문을 입력하면 관련 문서들을 검색하여 보여줍니다.

## 📝 주요 파일 설명

### `ingest_langchain.py`

RAG 시스템의 **데이터 준비 단계**를 담당합니다.

- `docs` 폴더의 문서를 읽어 LangChain Document 형식으로 변환
- 긴 문서를 작은 청크로 분할 (`RecursiveCharacterTextSplitter` 사용)
- 각 청크를 OpenAI 임베딩 모델로 벡터화
- ChromaDB에 저장

### `query_test.py`

RAG 시스템의 **검색(Retrieval) 단계**만 테스트하는 스크립트입니다.

- 저장된 ChromaDB를 로드
- 사용자 질문을 입력받음
- 벡터 유사도 검색으로 관련 문서 찾기
- 검색 결과 출력

### `config.py`

프로젝트 전체에서 사용하는 설정값을 관리합니다.

⚠️ **보안 주의사항**:
- 이 파일에는 OpenAI API 키가 포함되어 있습니다
- Git에 커밋하지 않도록 `.gitignore`에 추가하세요
- 실제 배포 시에는 환경 변수나 비밀 관리 서비스를 사용하세요

## ⚙️ 설정 옵션

`ingest_langchain.py`에서 조정 가능한 설정:

- `CHUNK_SIZE`: 청크 크기 (기본: 1500자)
- `CHUNK_OVERLAP`: 청크 간 겹치는 영역 (기본: 150자)
- `COLLECTION_NAME`: ChromaDB 컬렉션 이름 (기본: "my_rag_docs")

`config.py`에서 조정 가능한 설정:

- `EMBED_MODEL`: 임베딩 모델 (기본: "text-embedding-3-small")
- `CHAT_MODEL`: 채팅 모델 (기본: "gpt-4o-mini")
- `TOP_K`: 검색할 문서 개수 (기본: 4)

## 🔒 보안 고려사항

1. **API 키 관리**: `config.py`에 직접 API 키를 작성하지 말고, 환경 변수나 `.env` 파일을 사용하세요
2. **Git 제외**: `config.py`를 `.gitignore`에 추가하여 버전 관리에서 제외하세요
3. **템플릿 파일**: 공유할 경우 `config.example.py` 같은 템플릿 파일을 만들어서 API 키 부분만 비워두는 것을 권장합니다

## 📚 참고 자료

프로젝트 내 `docs` 폴더에는 RAG 개념, 기술 스택, 프로젝트 개요 등에 대한 문서가 포함되어 있습니다.

## 📄 라이선스

이 프로젝트는 학습용 프로젝트입니다.

