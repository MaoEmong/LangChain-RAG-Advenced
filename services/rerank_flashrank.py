"""
services/rerank_flashrank.py
============================================================
FlashRank Re-ranking 서비스

FlashRank 라이브러리를 사용하여 검색 결과를 재정렬합니다.
벡터 검색 결과를 질의와의 관련성에 따라 더 정확하게 재정렬합니다.

특징:
- FlashRank: 빠른 재정렬 라이브러리
- 질의와 문서의 의미적 관련성 기반 재정렬
- 검색 정확도 향상
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

# flashrank 라이브러리 (pip install flashrank)
from flashrank import Ranker, RerankRequest


class FlashRankReranker:
    """
    FlashRank Re-ranking 엔진 래퍼 클래스
    
    LangChain의 FlashrankRerank를 사용하지 않고
    flashrank 라이브러리를 직접 사용하여 재정렬을 수행합니다.
    """

    def __init__(self, model: str = "ms-marco-MiniLM-L-12-v2"):
        """
        FlashRankReranker 초기화
        
        Args:
            model: FlashRank 모델 이름 (기본값: "ms-marco-MiniLM-L-12-v2")
                - 다른 모델도 사용 가능 (FlashRank 문서 참고)
        """
        self._ranker = Ranker(model_name=model)

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """
        문서 리스트를 질의에 대한 관련성 순으로 재정렬합니다.
        
        FlashRank를 사용하여 질의와 각 문서의 의미적 관련성을 계산하고,
        관련성이 높은 순서로 문서를 재정렬합니다.
        
        Args:
            query: 검색 쿼리 문자열
            docs: 재정렬할 Document 리스트
        
        Returns:
            List[Document]: 재정렬된 Document 리스트
                - 관련성이 높은 순서로 정렬됨
        
        Note:
            - 빈 텍스트 문서는 최소 길이(공백 1개)로 보장
            - FlashRank는 텍스트가 너무 비면 오류가 발생하거나 품질이 떨어질 수 있음
            - 누락된 문서가 있으면 원래 순서 유지
        """
        if not docs:
            return []

        # FlashRank 입력 형식으로 변환
        passages = []
        for i, d in enumerate(docs):
            txt = (d.page_content or "").strip()
            # FlashRank는 텍스트가 너무 비면 오류/품질이 떨어질 수 있음
            if not txt:
                txt = " "  # 최소 길이 보장
            passages.append({"id": i, "text": txt})

        # FlashRank 재정렬 요청
        req = RerankRequest(query=query, passages=passages)
        ranked = self._ranker.rerank(req)

        # 재정렬 결과에서 ID 추출
        # ranked: [{"id": int, "score": float, "text": str}, ...] 형태
        ranked_ids = [item["id"] for item in ranked if "id" in item]

        # 혹시 누락된 ID가 있으면 뒤에 붙여서 안정적으로 반환
        seen = set(ranked_ids)
        for i in range(len(docs)):
            if i not in seen:
                ranked_ids.append(i)

        # 재정렬된 순서로 Document 리스트 반환
        return [docs[i] for i in ranked_ids]
