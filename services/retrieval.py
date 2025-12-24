"""
services/retrieval.py
============================================================
통합 검색 및 Re-Ranking 모듈

이 모듈은 2단계 검색 파이프라인을 제공합니다:
1. 벡터 유사도 검색으로 넓은 후보 확보 (initial_k)
2. FlashRank Re-Ranker로 관련성 높은 문서 재정렬 (top_k)

이 방식의 장점:
- 벡터 검색의 빠른 속도 + Re-Ranker의 높은 정확도
- Guardrail과 Confidence 계산을 위해 원본 distance score 보존
"""

from typing import List, Tuple

# Document와 distance score의 튜플 타입
# distance는 낮을수록 유사 (ChromaDB 기준)
DocumentScore = Tuple[object, float]  # (Document, distance_score)


def _doc_key(d) -> str:
    """
    Document를 고유하게 식별하기 위한 키를 생성합니다.
    
    rerank 후 원본 distance score를 매핑하기 위해 사용됩니다.
    source(파일 경로) + page_content 일부를 조합하여 안정적인 키를 만듭니다.
    
    Args:
        d: LangChain Document 객체
    
    Returns:
        str: Document를 식별하는 고유 키
    
    Note:
        - source와 page_content의 처음 200자를 조합
        - 같은 문서의 같은 청크는 항상 같은 키를 반환
    """
    src = (d.metadata or {}).get("source", "unknown")
    text = (d.page_content or "")
    return f"{src}::{text[:200]}"


def retrieve_with_rerank(
    vector_db,
    query: str,
    initial_k: int,
    top_k: int,
    reranker,
) -> List[DocumentScore]:
    """
    통합 검색 및 Re-Ranking 함수
    
    검색 파이프라인:
    1. 벡터 유사도 검색으로 넓은 후보 확보 (initial_k개)
    2. FlashRank Re-Ranker로 관련성 높은 문서 재정렬
    3. 상위 top_k개만 선택
    4. 원본 distance score를 rerank된 문서에 매핑
    
    Args:
        vector_db: ChromaDB 벡터 저장소 객체
        query (str): 검색 질문
        initial_k (int): 초기 후보 문서 개수 (넓게 가져올 개수)
        top_k (int): 최종 반환할 문서 개수
        reranker: FlashRankReranker 객체
    
    Returns:
        List[DocumentScore]: (Document, distance_score) 튜플 리스트
            - rerank된 순서로 정렬됨
            - 원본 벡터 검색의 distance score가 보존됨
    
    Note:
        - initial_k는 top_k보다 크게 설정하는 것이 좋음 (예: 20 vs 4)
        - rerank는 doc만 재정렬하므로 원본 score를 별도로 매핑해야 함
        - 매핑 실패 시 999.0 (불리한 값)을 부여하여 Guardrail에서 차단되도록 함
    """
    # ============================================================
    # 1단계: 벡터 유사도 검색으로 넓은 후보 확보
    # ============================================================
    # initial_k개를 가져와서 rerank할 풀을 만듦
    # score를 포함하여 가져옴 (Guardrail/Confidence 계산에 필요)
    candidates: List[DocumentScore] = vector_db.similarity_search_with_score(query, k=initial_k)

    # 검색 결과가 없으면 빈 리스트 반환
    if not candidates:
        return []

    # ============================================================
    # 2단계: FlashRank Re-Ranker로 문서 재정렬
    # ============================================================
    # FlashRank는 Document 리스트만 받으므로 doc만 추출
    docs = [d for d, _ in candidates]
    
    # Re-Ranker가 query 기준으로 관련성 높은 순서로 재정렬
    # top_n개만 반환 (최종적으로 필요한 개수)
    reranked_docs = reranker.rerank(query=query, docs=docs, top_n=top_k)

    # ============================================================
    # 3단계: 원본 distance score를 rerank된 문서에 매핑
    # ============================================================
    # 원본 후보들의 score를 키-값 맵으로 구성
    # _doc_key를 사용하여 Document를 고유하게 식별
    score_map = {_doc_key(d): float(s) for d, s in candidates}

    # rerank된 문서에 대해 원본 score를 매핑
    results: List[DocumentScore] = []
    for d in reranked_docs:
        s = score_map.get(_doc_key(d), None)
        # 매핑 실패 시 안전하게 큰 값(불리한 값) 부여
        # Guardrail에서 차단되도록 함
        results.append((d, float(s) if s is not None else 999.0))

    return results
