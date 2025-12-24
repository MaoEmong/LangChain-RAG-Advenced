# ============================================================
# ingest_langchain.py
#
# 이 파일의 역할 (RAG에서 매우 중요)
# ------------------------------------------------------------
# ✔ docs 폴더 안의 다양한 문서 파일을 읽는다
# ✔ 모든 파일을 LangChain의 Document 형태로 통일한다
# ✔ 긴 문서를 작은 chunk로 쪼갠다
# ✔ 각 chunk를 벡터로 변환해서 Chroma 벡터DB에 저장한다
#
# 즉,
# 👉 "RAG에서 검색할 수 있는 데이터"를 미리 만들어두는 단계
#
# 실행:
#   python ingest_langchain.py
#
# 주의:
# - 서버 코드가 아님
# - API 코드가 아님
# - 문서가 바뀌었을 때만 실행하면 됨
# ============================================================


# ----------------------------
# 파이썬 기본 라이브러리
# ----------------------------
import os      # 경로 처리, 폴더 생성
import glob    # 폴더 안 파일을 재귀적으로 검색


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


# ----------------------------
# 프로젝트 공통 설정
# ----------------------------
# config.py에 정의된 값들
from config import EMBED_MODEL, OPENAI_API_KEY


# ============================================================
# 이 파일 전용 설정값
# ============================================================

# 문서가 들어있는 폴더
DOCS_DIR = "./docs"

# 벡터DB가 저장될 폴더
CHROMA_DIR = "./chroma_db"

# Chroma 내부에서 사용하는 컬렉션 이름
COLLECTION_NAME = "my_rag_docs"

# chunk 크기
# - 너무 크면 검색이 둔해짐
# - 너무 작으면 문맥이 끊김
CHUNK_SIZE = 600

# chunk 겹침 영역
# - 앞/뒤 문맥이 자연스럽게 이어지도록 일부 겹침
CHUNK_OVERLAP = 100


# ============================================================
# 1️⃣ 문서 로딩 단계 (로더 확장 버전)
# ============================================================
def load_docs_from_folder(folder: str) -> list[Document]:
    """
    docs 폴더 안의 파일들을 확장자별 로더로 읽어서
    LangChain Document 리스트로 변환한다.

    지원 확장자:
    - .txt
    - .md
    - .pdf
    - .docx
    - .html / .htm
    """

    docs: list[Document] = []

    # (확장자, 로더 생성 함수) 매핑
    # 새로운 파일 타입을 추가하고 싶으면
    # 여기 한 줄만 추가하면 됨
    loader_rules = [
        (".txt",  lambda p: TextLoader(p, encoding="utf-8")),
        (".md",   lambda p: TextLoader(p, encoding="utf-8")),
        (".pdf",  lambda p: PyPDFLoader(p)),
        (".docx", lambda p: Docx2txtLoader(p)),
        (".html", lambda p: BSHTMLLoader(p)),
        (".htm",  lambda p: BSHTMLLoader(p)),
    ]

    # docs 폴더 아래 모든 파일을 재귀적으로 탐색
    for path in glob.glob(os.path.join(folder, "**/*"), recursive=True):

        # 파일이 아니면(폴더면) 무시
        if not os.path.isfile(path):
            continue

        # 확장자 추출 (.pdf, .txt 등)
        ext = os.path.splitext(path)[1].lower()

        # 확장자에 맞는 로더 찾기
        for rule_ext, make_loader in loader_rules:
            if ext == rule_ext:
                try:
                    # 로더 생성
                    loader = make_loader(path)

                    # 파일을 읽어서 Document 리스트 생성
                    loaded_docs = loader.load()

                    # source 메타데이터를 "파일 경로"로 통일
                    abs_path = os.path.abspath(path)
                    for d in loaded_docs:
                        d.metadata["source"] = abs_path

                    # 결과 누적
                    docs.extend(loaded_docs)

                except Exception as e:
                    # 파일 하나가 깨져 있어도 전체 ingest가 멈추지 않게 함
                    print(f"[WARN] failed to load: {path} ({e})")

                # 로더 찾았으면 다음 파일로
                break

    return docs


# ============================================================
# 2️⃣ 청킹 단계
# ============================================================
def chunk_docs(docs: list[Document]) -> list[Document]:
    """
    긴 Document들을 작은 chunk Document들로 쪼갠다
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # 입력  : [Document, Document, ...]
    # 출력  : [chunked Document, chunked Document, ...]
    return splitter.split_documents(docs)


# ============================================================
# 3️⃣ 벡터DB 저장 단계
# ============================================================
def build_or_update_chroma(chunks: list[Document]) -> None:
    """
    chunk Document들을 임베딩해서
    Chroma 벡터DB에 저장한다
    """

    # 임베딩 객체 생성
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=OPENAI_API_KEY
    )

    # Chroma 벡터DB 로드 또는 생성
    # persist_directory에 자동으로 파일 저장됨
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # chunk Document들을 그대로 DB에 추가
    db.add_documents(chunks)


# ============================================================
# 메인 실행 함수
# ============================================================
def main():
    """
    ingest_langchain.py 실행 시
    여기부터 시작된다
    """

    # docs / chroma_db 폴더가 없으면 생성
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    # 1) 문서 로딩
    docs = load_docs_from_folder(DOCS_DIR)
    if not docs:
        print(f"[WARN] no docs found in {DOCS_DIR}")
        return

    # 2) 청킹
    chunks = chunk_docs(docs)

    # 3) 벡터DB 저장
    build_or_update_chroma(chunks)

    print(f"[OK] loaded docs: {len(docs)}, stored chunks: {len(chunks)}")


# ============================================================
# 파이썬 파일 직접 실행 시 시작 지점
# ============================================================
if __name__ == "__main__":
    main()
