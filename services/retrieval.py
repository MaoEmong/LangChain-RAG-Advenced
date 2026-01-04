"""
services/retrieval.py
============================================================
문서 검색 서비스 (Parent-Child 기반)

Parent-Child 문서 구조를 사용한 고급 검색 기능을 제공합니다.
- Child(청크)로 검색하여 정확도 향상
- Parent(원문)로 복원하여 컨텍스트 제공
- 키워드 바이어스로 OCR 문서 검색 최적화
- Re-ranking으로 검색 결과 재정렬

주요 기능:
1. 키워드 바이어스: OCR/스캔 문서 검색 시 키워드 매칭 우선
2. Child 검색: 벡터 DB에서 작은 청크 단위로 검색
3. Parent 복원: 검색된 청크의 원문 문서 복원
4. Re-ranking: FlashRank를 사용한 검색 결과 재정렬
"""

from __future__ import annotations

import re
from typing import List, Tuple, Dict

from langchain_core.documents import Document

DocumentScore = Tuple[Document, float]  # (Document, distance_score) 튜플 타입


# =========================
# 1) Keyword-bias heuristics (키워드 바이어스 휴리스틱)
# =========================
# 운송장 번호, 트래킹 코드 패턴 정규식 (예: APX3002345386815CN)
_TRACKING_RE = re.compile(r"\b[A-Z]{2,6}\d{8,20}[A-Z]{0,4}\b", re.IGNORECASE)

def _looks_like_ocr_keyword_query(q: str) -> bool:
    """
    OCR/스캔 문서에서 키워드 매칭이 특히 중요한 질의인지 판별합니다.
    
    운송장 번호, 트래킹 코드, 송장 번호 등의 키워드 검색은
    OCR 문서에서 정확한 문자열 매칭이 중요하므로
    OCR 문서를 우선 검색하도록 바이어스를 적용합니다.
    
    Args:
        q: 사용자 질의 문자열
    
    Returns:
        bool: OCR 키워드 질의로 판별되면 True
    
    판별 기준:
        - 운송장, 송장, 트래킹 등의 키워드 포함
        - 운송장 번호 패턴 (대문자+숫자 조합) 매칭
        - 긴 영문+숫자 조합 (10자 이상)
    """
    s = (q or "").strip()
    if not s:
        return False

    up = s.upper()

    # 흔한 키워드들 (운송장, 트래킹 관련)
    keywords = [
        "운송장", "송장", "트래킹", "배송", "배송조회", "운송", "택배",
        "INVOICE", "TRACK", "TRACKING", "WAYBILL",
        "APX", "4PX", "DHL", "FEDEX", "UPS",
    ]
    if any(k in up for k in keywords):
        return True

    # 트래킹 코드처럼 보이는 패턴 (예: APX3002345386815CN)
    if _TRACKING_RE.search(up):
        return True

    # 영문+숫자 조합이 길게 들어오면 OCR 키워드일 가능성 높음
    alnum = re.sub(r"[^A-Z0-9]", "", up)
    if len(alnum) >= 10 and re.search(r"[A-Z]", alnum) and re.search(r"\d", alnum):
        return True

    return False


def _search_children(vector_db, query: str, k: int, flt: Dict | None = None) -> List[Tuple[Document, float]]:
    """
    벡터 DB에서 child 문서(청크)를 검색합니다.
    
    ChromaDB의 similarity_search_with_score를 사용하여
    유사도 점수와 함께 문서를 검색합니다.
    
    Args:
        vector_db: ChromaDB 벡터 저장소
        query: 검색 쿼리 문자열
        k: 반환할 문서 개수
        flt: 메타데이터 필터 딕셔너리 (선택사항)
            - 예: {"domain": "ocr_scan"} → OCR 스캔 문서만 검색
    
    Returns:
        List[Tuple[Document, float]]: (Document, distance_score) 튜플 리스트
            - distance_score가 낮을수록 유사도 높음
    
    Note:
        - langchain-chroma는 similarity_search_with_score 메서드를 제공합니다
        - filter는 메타데이터 기반 where 조건으로 적용됩니다
    """
    if flt:
        # 메타데이터 필터 적용 검색
        return vector_db.similarity_search_with_score(query, k=k, filter=flt)
    # 필터 없이 검색
    return vector_db.similarity_search_with_score(query, k=k)


def _get_child_candidates(vector_db, query: str, initial_k: int) -> List[Tuple[Document, float]]:
    """
    Child 문서(청크) 후보를 검색합니다 (키워드 바이어스 적용).
    
    OCR/스캔 키워드 질의인 경우:
    1. 먼저 OCR 스캔 문서만 검색 (domain=ocr_scan)
    2. 결과가 충분하면 그대로 사용
    3. 부족하면 전체 검색으로 보강 (중복 제거)
    
    일반 질의인 경우:
    - 전체 문서에서 검색
    
    Args:
        vector_db: ChromaDB 벡터 저장소
        query: 검색 쿼리 문자열
        initial_k: 초기 검색 개수
    
    Returns:
        List[Tuple[Document, float]]: (Document, distance_score) 튜플 리스트
            - distance 오름차순 정렬 (best first)
    """
    use_ocr_bias = _looks_like_ocr_keyword_query(query)

    if use_ocr_bias:
        # 키워드 바이어스 적용: OCR 스캔 문서 우선 검색
        # 1) OCR 스캔 문서만 먼저 검색
        ocr_only = _search_children(vector_db, query, k=initial_k, flt={"domain": "ocr_scan"})
        
        # 충분한 결과가 있으면 그대로 사용
        if len(ocr_only) >= max(3, initial_k // 4):
            return ocr_only

        # 2) 부족하면 전체 검색으로 보강 (중복 제거)
        all_docs = _search_children(vector_db, query, k=initial_k, flt=None)

        # 중복 제거를 위한 키 생성 (source, doc_id, content 시작 부분)
        seen = set()
        merged: List[Tuple[Document, float]] = []

        for d, s in ocr_only + all_docs:
            key = (d.metadata.get("source"), d.metadata.get("doc_id"), d.page_content[:80])
            if key in seen:
                continue
            seen.add(key)
            merged.append((d, float(s)))

        # distance 오름차순 정렬 (best first)
        merged.sort(key=lambda x: x[1])
        return merged[:initial_k]

    # 일반 질의: 기존 방식 (전체 검색)
    return _search_children(vector_db, query, k=initial_k, flt=None)


def _restore_parents(docstore, child_scored: List[Tuple[Document, float]], parent_id_key: str) -> List[DocumentScore]:
    """
    Child 문서(청크) 결과에서 Parent 문서(원문)를 복원합니다.
    
    Child 문서의 doc_id를 기반으로 Parent 문서를 조회하고,
    같은 Parent의 여러 Child 중 가장 좋은 점수를 Parent의 대표 점수로 사용합니다.
    
    Args:
        docstore: Parent 문서 저장소 (SQLiteDocStore)
        child_scored: Child 문서와 점수 튜플 리스트
        parent_id_key: 메타데이터에서 Parent ID를 가져올 키 이름 (기본: "doc_id")
    
    Returns:
        List[DocumentScore]: (Parent Document, best_score) 튜플 리스트
            - score 오름차순 정렬 (best first)
    
    Note:
        - 같은 Parent의 여러 Child가 있으면 가장 좋은 점수(최소 distance)를 사용
        - Distance 점수는 낮을수록 유사도가 높으므로 좋은 점수입니다
    """
    best_score_by_parent: Dict[str, float] = {}  # Parent별 최고 점수
    parent_doc_by_id: Dict[str, Document] = {}   # Parent 문서 캐시

    # Child 문서를 순회하며 Parent별 최고 점수 찾기
    for child_doc, score in child_scored:
        pid = child_doc.metadata.get(parent_id_key)
        if not pid:
            continue

        # Parent별 best score 유지 (distance는 낮을수록 좋음)
        if (pid not in best_score_by_parent) or (score < best_score_by_parent[pid]):
            best_score_by_parent[pid] = float(score)

    if not best_score_by_parent:
        return []

    # Parent 문서 조회
    parent_ids = list(best_score_by_parent.keys())
    parent_docs = docstore.mget(parent_ids)  # None 포함 가능

    # Parent 문서 캐시 구축
    for pid, pdoc in zip(parent_ids, parent_docs):
        if pdoc:
            parent_doc_by_id[pid] = pdoc

    # 결과 리스트 생성
    results: List[DocumentScore] = []
    for pid, score in best_score_by_parent.items():
        pdoc = parent_doc_by_id.get(pid)
        if pdoc:
            results.append((pdoc, float(score)))

    # score 오름차순 정렬 (best first)
    results.sort(key=lambda x: x[1])
    return results


def retrieve_parents_with_rerank(
    vector_db,
    docstore,
    query: str,
    initial_k: int,
    top_k: int,
    reranker,
    parent_id_key: str = "doc_id",
) -> List[DocumentScore]:
    """
    Parent-Child 구조를 사용한 문서 검색 (Re-ranking 포함).
    
    검색 프로세스:
    1. Child 벡터 검색: initial_k 개의 Child 문서(청크) 검색 (키워드 바이어스 포함)
    2. Re-ranking: FlashRank를 사용하여 Child 문서 재정렬
    3. Parent 복원: 상위 top_k 개의 Child에 해당하는 Parent 문서 복원
    
    Args:
        vector_db: ChromaDB 벡터 저장소
        docstore: Parent 문서 저장소 (SQLiteDocStore)
        query: 검색 쿼리 문자열
        initial_k: 초기 검색 개수 (Child 문서)
        top_k: 최종 반환 개수 (Parent 문서)
        reranker: Re-ranking 엔진 (FlashRankReranker 등)
        parent_id_key: 메타데이터에서 Parent ID를 가져올 키 이름
    
    Returns:
        List[DocumentScore]: (Parent Document, score) 튜플 리스트
            - 최대 top_k 개까지 반환
            - score 오름차순 정렬 (best first)
    
    Note:
        - 키워드 바이어스: OCR 키워드 질의 시 OCR 문서 우선 검색
        - Re-ranking으로 검색 정확도 향상
        - Parent 복원으로 원문 컨텍스트 제공
    """
    # 1) Child 후보 검색 (키워드 바이어스 포함)
    child_scored = _get_child_candidates(vector_db, query, initial_k)
    if not child_scored:
        return []

    child_docs = [d for d, _ in child_scored]

    # 2) Re-ranking: FlashRank로 Child 문서 재정렬
    reranked_docs = reranker.rerank(query, child_docs)

    # Re-ranking 결과에 따라 Child 점수 리스트 재정렬
    # reranker가 Document 객체를 그대로 반환한다고 가정
    reranked_set = {id(d): i for i, d in enumerate(reranked_docs)}
    child_scored.sort(key=lambda pair: reranked_set.get(id(pair[0]), 10**9))

    # 3) 상위 top_k 개의 Child만 사용하여 Parent 복원
    child_scored = child_scored[: max(top_k, 1)]
    parents = _restore_parents(docstore, child_scored, parent_id_key=parent_id_key)
    return parents[:top_k]
