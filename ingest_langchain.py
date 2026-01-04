"""
ingest_langchain.py
============================================================
PDF / TXT / MD 문서를
- 파일당 parent 1개
- child chunk는 Chroma에 batch 저장
- parent는 SQLiteDocStore 저장
"""

# ----------------------------
# 기본 라이브러리
# ----------------------------
import os
import uuid
from typing import List

# ----------------------------
# LangChain
# ----------------------------
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ----------------------------
# 프로젝트 모듈
# ----------------------------
from loaders.auto_loader import load_docs_from_folder
from docstore_sqlite import SQLiteDocStore
from config import (
    DOCSTORE_PATH,
    EMBED_MODEL,
    OPENAI_API_KEY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# ----------------------------
# 경로 설정
# ----------------------------
DOCS_DIR = "./docs"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "my_rag_docs"

# ============================================================
# Core ingest logic
# ============================================================
def ingest_file_level_parent_docs(docs: List[Document]) -> None:
    """
    파일 단위 Parent 문서를 생성하여 저장합니다.
    
    처리 프로세스:
    1. 문서를 doc_id 기준으로 그룹화 (파일 단위)
    2. 각 파일의 모든 페이지를 하나의 Parent 문서로 통합
    3. Parent 문서를 SQLiteDocStore에 저장
    4. Parent 문서를 청크로 분할하여 Child 문서 생성
    5. Child 문서를 ChromaDB에 벡터로 저장
    
    Args:
        docs: load_docs_from_folder()로 로드한 문서 리스트
            - 같은 파일의 문서는 같은 doc_id를 가집니다
    
    Note:
        - 파일당 Parent 문서 1개를 생성합니다
        - Child 문서는 배치 단위로 저장하여 성능 최적화
    """

    # 1) 데이터베이스 초기화
    # OpenAI 임베딩 모델 초기화 (텍스트를 벡터로 변환)
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=OPENAI_API_KEY,
    )

    # ChromaDB 벡터 저장소 초기화 (Child 문서 저장용)
    chroma = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # SQLite 문서 저장소 초기화 (Parent 문서 저장용)
    docstore = SQLiteDocStore(DOCSTORE_PATH)

    # 2) 텍스트 분할기 초기화 (Parent → Child 분할)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,      # 청크 크기
        chunk_overlap=CHUNK_OVERLAP,  # 청크 간 겹치는 부분
    )

    # 3) doc_id 기준으로 파일 단위 Parent 묶기
    # 같은 파일의 모든 페이지를 하나의 그룹으로 묶음
    parents_by_id: dict[str, list[Document]] = {}
    for d in docs:
        doc_id = d.metadata["doc_id"]
        parents_by_id.setdefault(doc_id, []).append(d)

    print(f"[INFO] parent files: {len(parents_by_id)}")

    total_children = 0

    # 4) Parent 문서 1개씩 처리
    for idx, (doc_id, pages) in enumerate(parents_by_id.items(), 1):
        # Parent = 파일 전체 텍스트 합치기 (모든 페이지 통합)
        full_text = "\n\n".join(p.page_content for p in pages)

        # Parent 문서 생성
        parent_doc = Document(
            page_content=full_text,
            metadata={
                "doc_id": doc_id,
                "source": pages[0].metadata.get("source"),
                "kind": pages[0].metadata.get("kind"),
            },
        )

        # Parent 문서 저장 (SQLite)
        docstore.mset([(doc_id, parent_doc)])

        # Child 문서로 분할 (Parent를 작은 청크로 나눔)
        child_docs = child_splitter.split_documents([parent_doc])

        # Child 문서에 메타데이터 추가
        for cd in child_docs:
            cd.metadata["doc_id"] = doc_id
            cd.metadata["source"] = parent_doc.metadata["source"]
            cd.metadata["kind"] = parent_doc.metadata["kind"]

        # 5) Child 문서를 배치로 저장 (ChromaDB 제한 회피)
        BATCH = 1500  # 배치 크기
        for i in range(0, len(child_docs), BATCH):
            batch = child_docs[i:i + BATCH]
            chroma.add_documents(batch)  # 벡터 변환 후 저장
            total_children += len(batch)

        print(
            f"[OK] parent {idx}/{len(parents_by_id)} | "
            f"children={len(child_docs)} | total_children={total_children}"
        )

    print("[DONE] ingest complete")

# ============================================================
# 메인 함수
# ============================================================
def main():
    """
    문서 수집(ingestion) 메인 함수
    
    처리 프로세스:
    1. 필요한 디렉토리 생성
    2. 폴더에서 모든 문서 로드
    3. 문서 종류 통계 출력
    4. Parent-Child 구조로 저장
    """
    # 필요한 디렉토리 생성
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    # 문서 로드
    docs = load_docs_from_folder(DOCS_DIR)
    if not docs:
        print("[WARN] no docs found")
        return

    # 문서 종류 통계
    kinds = {}
    for d in docs:
        kinds[d.metadata.get("kind", "?")] = kinds.get(d.metadata.get("kind", "?"), 0) + 1
    print("[INFO] kinds:", kinds)

    # Parent-Child 구조로 저장
    ingest_file_level_parent_docs(docs)

if __name__ == "__main__":
    main()
