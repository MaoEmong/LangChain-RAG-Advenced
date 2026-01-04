"""
query_test.py
============================================================
ParentDocumentRetriever 동작 테스트

LangChain의 ParentDocumentRetriever를 사용한 문서 검색 테스트 스크립트입니다.

테스트 내용:
- Child(Chroma)로 검색 → Parent(SQLite)로 복원
- 대화형 질의응답 테스트

참고:
- LangChain v1.x + langchain-classic 기준
"""

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever

from docstore_sqlite import SQLiteDocStore
from config import (
    OPENAI_API_KEY,
    EMBED_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
    DOCSTORE_PATH,
)

# ingest_langchain.py에 맞춤 (현재 파일 내부 상수로 사용)
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

def build_parent_retriever():
    """
    ParentDocumentRetriever를 구성하고 반환합니다.
    
    Returns:
        ParentDocumentRetriever: LangChain ParentDocumentRetriever 객체
    """
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)

    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

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

def main():
    """
    대화형 질의응답 테스트 메인 함수
    
    사용자 입력을 받아 검색하고 결과를 출력합니다.
    'quit' 또는 Ctrl+C로 종료할 수 있습니다.
    """
    retriever = build_parent_retriever()

    while True:
        q = input("질문> ").strip()
        if not q:
            continue

        # LangChain v1.x 계열에서는 invoke()가 표준 메서드
        docs = retriever.invoke(q)

        print("\n--- TOP MATCHES (PARENTS expected) ---")
        for i, d in enumerate(docs[:4], start=1):
            src = (d.metadata or {}).get("source", "unknown")
            print(f"\n[{i}] source={src}  len={len(d.page_content)}")
            print(d.page_content[:800])

if __name__ == "__main__":
    main()
