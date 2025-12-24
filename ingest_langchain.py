"""
ingest_langchain.py
============================================================
문서 수집 및 벡터DB 구축 스크립트

이 파일의 역할 (RAG에서 매우 중요):
------------------------------------------------------------
✔ docs 폴더 안의 다양한 문서 파일을 읽는다
✔ 모든 파일을 LangChain의 Document 형태로 통일한다
✔ 긴 문서를 작은 chunk로 쪼갠다
✔ 각 chunk를 벡터로 변환해서 Chroma 벡터DB에 저장한다

즉,
👉 "RAG에서 검색할 수 있는 데이터"를 미리 만들어두는 단계

실행:
  python ingest_langchain.py

주의:
- 서버 코드가 아님 (독립 실행 스크립트)
- API 코드가 아님
- 문서가 바뀌었을 때만 실행하면 됨
- 실행 후 벡터DB가 생성/업데이트됨
"""


# ----------------------------
# 파이썬 기본 라이브러리
# ----------------------------
import os      # 경로 처리, 폴더 생성
import glob    # 폴더 안 파일을 재귀적으로 검색
import uuid


# ----------------------------
# LangChain 핵심 자료구조
# ----------------------------

# Document:
# - LangChain에서 사용하는 "문서 표준 형태"
# - page_content : 실제 텍스트
# - metadata     : 출처, 페이지 번호, 기타 정보
from langchain_core.documents import Document


# RecursiveCharacterTextSplitter:
# - 긴 텍스트를 자연스럽게 작은 단위(chunk)로 쪼개는 도구
from langchain_text_splitters import RecursiveCharacterTextSplitter


# OpenAIEmbeddings:
# - 텍스트 → 숫자 벡터(임베딩)로 변환
from langchain_openai import OpenAIEmbeddings


# Chroma:
# - 로컬 파일 기반 벡터DB
from langchain_chroma import Chroma


# ----------------------------
# 문서 로더들 (파일 타입별)
# ----------------------------
# 각 파일을 읽어서 Document 리스트로 만들어주는 역할
from langchain_community.document_loaders import (
    TextLoader,                 # .txt
    UnstructuredMarkdownLoader, # .md
    PyPDFLoader,                # .pdf
    Docx2txtLoader,             # .docx
    BSHTMLLoader,               # .html / .htm
)

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from docstore_sqlite import SQLiteDocStore
from config import DOCSTORE_PATH

# ----------------------------
# 프로젝트 공통 설정
# ----------------------------
# config.py에 정의된 값들
from config import EMBED_MODEL, OPENAI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP


# ============================================================
# 이 파일 전용 설정값
# ============================================================

# 문서가 들어있는 폴더
DOCS_DIR = "./docs"

# 벡터DB가 저장될 폴더
CHROMA_DIR = "./chroma_db"

# Chroma 내부에서 사용하는 컬렉션 이름
COLLECTION_NAME = "my_rag_docs"

# ============================================================
# 1️⃣ 문서 로딩 단계 (로더 확장 버전)
# ============================================================
def load_docs_from_folder(folder: str) -> list[Document]:
    """
    docs 폴더 안의 파일들을 확장자별 로더로 읽어서
    LangChain Document 리스트로 변환합니다.

    이 함수는:
    1. 지정된 폴더를 재귀적으로 탐색
    2. 파일 확장자에 맞는 로더 선택
    3. 각 파일을 Document 객체로 변환
    4. 메타데이터에 출처 정보 추가

    지원 확장자:
    - .txt: 텍스트 파일
    - .md: 마크다운 파일
    - .pdf: PDF 문서
    - .docx: Word 문서
    - .html / .htm: HTML 파일

    Args:
        folder (str): 문서가 있는 폴더 경로

    Returns:
        list[Document]: 변환된 Document 객체 리스트

    Note:
        - 파일 로딩 실패 시 경고만 출력하고 계속 진행
        - 새로운 파일 타입 추가 시 loader_rules에 추가하면 됨
    """
    docs: list[Document] = []

    # (확장자, 로더 생성 함수) 매핑
    # 새로운 파일 타입을 추가하고 싶으면 여기 한 줄만 추가하면 됨
    loader_rules = [
        (".txt",  lambda p: TextLoader(p, encoding="utf-8")),
        (".md",   lambda p: TextLoader(p, encoding="utf-8")),
        (".pdf",  lambda p: PyPDFLoader(p)),
        (".docx", lambda p: Docx2txtLoader(p)),
        (".html", lambda p: BSHTMLLoader(p)),
        (".htm",  lambda p: BSHTMLLoader(p)),
    ]

    # docs 폴더 아래 모든 파일을 재귀적으로 탐색
    # **/* 패턴으로 하위 폴더까지 모두 탐색
    for path in glob.glob(os.path.join(folder, "**/*"), recursive=True):

        # 파일이 아니면(폴더면) 무시
        if not os.path.isfile(path):
            continue

        # 확장자 추출 (.pdf, .txt 등) - 소문자로 변환하여 대소문자 구분 없이 매칭
        ext = os.path.splitext(path)[1].lower()

        # 확장자에 맞는 로더 찾기
        for rule_ext, make_loader in loader_rules:
            if ext == rule_ext:
                try:
                    # 로더 생성 (파일 타입별로 적절한 로더 사용)
                    loader = make_loader(path)

                    # 파일을 읽어서 Document 리스트 생성
                    # PDF는 페이지별로, 텍스트는 전체로 Document 생성
                    loaded_docs = loader.load()

                    # source 메타데이터를 "파일 경로"로 통일
                    # 절대 경로로 변환하여 일관성 유지
                    abs_path = os.path.abspath(path)
                    for d in loaded_docs:
                        d.metadata["source"] = abs_path

                    # 결과 누적
                    docs.extend(loaded_docs)

                except Exception as e:
                    # 파일 하나가 깨져 있어도 전체 ingest가 멈추지 않게 함
                    # (예: 암호화된 PDF, 손상된 파일 등)
                    print(f"[WARN] failed to load: {path} ({e})")

                # 로더 찾았으면 다음 파일로 이동
                break

    return docs


# ============================================================
# 2️⃣ 청킹 단계
# ============================================================
def chunk_docs(docs: list[Document]) -> list[Document]:
    """
    긴 Document들을 작은 chunk Document들로 쪼갭니다.

    RAG 시스템에서 긴 문서를 작은 단위로 나누는 이유:
    1. 검색 정확도 향상: 관련 부분만 정확히 찾을 수 있음
    2. 컨텍스트 길이 제한: LLM의 토큰 제한 내에서 처리 가능
    3. 효율성: 필요한 부분만 검색하여 비용 절감

    Args:
        docs (list[Document]): 원본 Document 리스트

    Returns:
        list[Document]: 청킹된 Document 리스트

    Note:
        - RecursiveCharacterTextSplitter는 문단, 문장, 단어 단위로
          자연스럽게 분할하여 문맥을 최대한 보존합니다
        - chunk_overlap을 사용하여 앞뒤 문맥이 끊기지 않도록 합니다
    """
    # RecursiveCharacterTextSplitter 초기화
    # - chunk_size: 각 청크의 최대 문자 수
    # - chunk_overlap: 청크 간 겹치는 문자 수 (문맥 보존)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # 입력  : [Document, Document, ...] (원본 문서들)
    # 출력  : [chunked Document, chunked Document, ...] (작은 청크들)
    return splitter.split_documents(docs)

def build_parent_retriever(db: Chroma) -> ParentDocumentRetriever:
    """Parent/Child 구조를 구성해주는 retriever 팩토리."""
    docstore = SQLiteDocStore(DOCSTORE_PATH)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return ParentDocumentRetriever(
        vectorstore=db,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        parent_id_key="doc_id",
    )

# ============================================================
# 3️⃣ 벡터DB 저장 단계
# ============================================================
def build_or_update_chroma(chunks: list[Document]) -> None:
    """
    chunk Document들을 임베딩해서 Chroma 벡터DB에 저장합니다.

    이 함수는:
    1. OpenAI 임베딩 모델 초기화
    2. ChromaDB 벡터 저장소 생성/로드
    3. 각 청크를 벡터로 변환하여 저장

    Args:
        chunks (list[Document]): 저장할 청크 Document 리스트

    Returns:
        None

    Note:
        - 이미 벡터DB가 있으면 기존 데이터에 추가됩니다
        - persist_directory에 자동으로 파일이 저장됩니다
        - 임베딩 변환은 OpenAI API를 호출하므로 시간이 걸릴 수 있습니다
    """
    # OpenAI 임베딩 객체 생성
    # 텍스트를 벡터(숫자 배열)로 변환하는 데 사용
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=OPENAI_API_KEY
    )

    # Chroma 벡터DB 로드 또는 생성
    # - collection_name: 저장소 내부의 컬렉션 이름
    # - embedding_function: 벡터 변환 함수
    # - persist_directory: 벡터DB 파일 저장 경로 (자동 저장됨)
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # chunk Document들을 벡터DB에 추가
    # 각 청크의 텍스트가 임베딩되어 벡터로 변환되고 저장됨
    db.add_documents(chunks)

def build_or_load_chroma() -> Chroma:
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return db


# ============================================================
# 메인 실행 함수
# ============================================================
def main():
    """
    ingest_langchain.py 실행 시 여기부터 시작됩니다.

    메인 실행 흐름:
    1. 필요한 폴더 생성
    2. 문서 로딩 (다양한 파일 형식 지원)
    3. 문서 청킹 (긴 문서를 작은 단위로 분할)
    4. 벡터DB 저장 (임베딩 변환 및 저장)

    Returns:
        None
    """
    # docs / chroma_db 폴더가 없으면 생성
    # exist_ok=True: 이미 존재해도 에러 발생 안 함
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    # 1단계: 문서 로딩
    # docs 폴더의 모든 문서를 LangChain Document로 변환
    docs = load_docs_from_folder(DOCS_DIR)
    if not docs:
        print(f"[WARN] no docs found in {DOCS_DIR}")
        return

    # ✅ parent_id(doc_id) 부여(필수)
    for d in docs:
        d.metadata["doc_id"] = str(uuid.uuid4())

    db = build_or_load_chroma()
    retriever = build_parent_retriever(db)

    # ✅ 핵심: parent는 SQLite에, child는 Chroma에 들어감
    retriever.add_documents(docs)

    print(f"[OK] loaded docs: {len(docs)} (parents stored in sqlite, children in chroma)")


# ============================================================
# 파이썬 파일 직접 실행 시 시작 지점
# ============================================================
if __name__ == "__main__":
    main()
