"""
services/rerank_flashrank.py
============================================================
FlashRank 기반 경량 Re-Ranker 래퍼

이 모듈은 FlashRank를 사용하여 검색 결과를 재정렬합니다.
FlashRank는 경량화된 cross-encoder 모델로, 벡터 검색 결과의 정확도를 향상시킵니다.

작동 방식:
1. 벡터 검색으로 넓은 후보 확보
2. FlashRank가 query와 각 문서의 관련성을 더 정확히 평가
3. 관련성 높은 순서로 재정렬하여 상위 문서만 반환

장점:
- 벡터 검색보다 정확한 관련성 평가
- 경량 모델로 빠른 처리 속도
- 추가 API 호출 없이 로컬에서 실행
"""

from typing import List

# LangChain 버전에 따라 import 경로가 달라질 수 있어 방어적으로 작성
try:
    from langchain_community.document_compressors import FlashrankRerank
except Exception:
    # 일부 버전에서 다른 경로인 경우가 있어 대비
    from langchain.retrievers.document_compressors import FlashrankRerank


class FlashRankReranker:
    """
    FlashRank Re-Ranker 래퍼 클래스
    
    LangChain의 FlashrankRerank를 래핑하여 간단한 인터페이스를 제공합니다.
    """
    
    def __init__(self, model: str = "ms-marco-MiniLM-L-12-v2"):
        """
        FlashRank Re-Ranker 초기화
        
        Args:
            model (str): 사용할 FlashRank 모델 이름
                - 기본값: "ms-marco-MiniLM-L-12-v2" (경량 모델)
                - 다른 모델: "ms-marco-MiniLM-L-6-v2" (더 작음), 
                            "rank-T5-flan" (더 정확하지만 느림)
        
        Note:
            - FlashrankRerank는 내부적으로 모델을 다운로드/캐시할 수 있음
            - 첫 실행 시 모델 다운로드로 시간이 걸릴 수 있음
        """
        # FlashrankRerank는 내부적으로 모델을 다운로드/캐시할 수 있음
        self._compressor = FlashrankRerank(model=model)

    def rerank(self, query: str, docs: List, top_n: int) -> List:
        """
        문서들을 query 기준으로 재정렬하여 상위 top_n개만 반환합니다.
        
        FlashRank는 cross-encoder 방식으로 query와 각 문서의 관련성을
        더 정확하게 평가하여 벡터 검색 결과를 개선합니다.
        
        Args:
            query (str): 검색 질문
            docs (List): 재정렬할 Document 리스트
            top_n (int): 반환할 상위 문서 개수
        
        Returns:
            List: 재정렬된 Document 리스트 (상위 top_n개)
        
        Note:
            - 빈 리스트 입력 시 빈 리스트 반환
            - FlashrankRerank.compress_documents()를 사용하여 재정렬
            - 반환된 문서는 관련성 높은 순서로 정렬됨
        """
        if not docs:
            return []
        
        # FlashrankRerank는 compress_documents(docs, query)를 제공
        # query와 각 문서의 관련성을 평가하여 재정렬
        reranked = self._compressor.compress_documents(docs, query=query)
        
        # 상위 top_n개만 반환
        return reranked[:top_n]
