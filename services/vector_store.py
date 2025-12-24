"""
services/vector_store.py
============================================================
벡터 저장소 생성 및 관리

이 모듈은 ChromaDB 벡터 저장소를 생성하고 반환합니다.
이미 저장된 벡터DB를 로드하거나, 새로 생성할 수 있습니다.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def create_vector_store(
    api_key,
    embed_model,
    persist_dir,
    collection_name,
):
    """
    ChromaDB 벡터 저장소를 생성하고 반환합니다.
    
    이 함수는:
    1. OpenAI 임베딩 모델을 초기화
    2. ChromaDB 벡터 저장소를 생성/로드
    3. 지정된 컬렉션에 연결
    
    Args:
        api_key (str): OpenAI API 키
        embed_model (str): 임베딩 모델 이름 (예: "text-embedding-3-small")
        persist_dir (str): 벡터DB 저장 디렉토리 경로
        collection_name (str): ChromaDB 컬렉션 이름
    
    Returns:
        Chroma: ChromaDB 벡터 저장소 객체
        
    Note:
        - persist_dir에 이미 벡터DB가 있으면 자동으로 로드됩니다
        - 없으면 새로 생성됩니다
        - ingest_langchain.py로 문서를 먼저 저장해야 합니다
    """
    # OpenAI 임베딩 모델 초기화
    # 텍스트를 벡터로 변환하는 데 사용
    embeddings = OpenAIEmbeddings(
        model=embed_model,
        api_key=api_key,
    )

    # ChromaDB 벡터 저장소 생성/로드
    # - collection_name: 저장소 내부의 컬렉션 이름
    # - embedding_function: 벡터 변환 함수 (OpenAI 임베딩)
    # - persist_directory: 벡터DB 파일 저장 경로
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
