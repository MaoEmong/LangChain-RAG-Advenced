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
from langchain_core.documents import Document
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
    docstore=None,           # ✅ 추가
    parent_id_key="doc_id",  # ✅ 추가
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
    candidates = vector_db.similarity_search_with_score(query, k=initial_k)
    if not candidates:
        return []
    
    # ✅ Parent 모드
    if docstore is not None:
        # 1) child → parent_id별 best score 추출
        best_score_by_pid = {}
        for d, s in candidates:
            pid = (d.metadata or {}).get(parent_id_key)
            if not pid:
                continue
            s = float(s)
            if pid not in best_score_by_pid or s < best_score_by_pid[pid]:
                best_score_by_pid[pid] = s
    
        if not best_score_by_pid:
            return []
    
        # 2) parent 문서 로드
        pids = list(best_score_by_pid.keys())
        parent_docs = docstore.mget(pids)
    
        # mget 결과엔 None이 섞일 수 있으니 필터
        parents = []
        for pid, pd in zip(pids, parent_docs):
            if pd is None:
                continue
            pd.metadata = pd.metadata or {}
            pd.metadata[parent_id_key] = pid  # parent에도 id 보장
            parents.append(pd)
    
        if not parents:
            return []
    
        # 3) parent로 rerank
        reranked_parents = reranker.rerank(query=query, docs=parents, top_n=top_k)
    
        # 4) rerank 결과에 score 매핑
        results = []
        for pd in reranked_parents:
            pid = (pd.metadata or {}).get(parent_id_key)
            results.append((pd, float(best_score_by_pid.get(pid, 999.0))))
        return results
    

def retrieve_parents_with_rerank(
    vector_db,
    docstore,                 # SQLiteDocStore
    query: str,
    initial_k: int,
    top_k: int,
    reranker,
    parent_id_key: str = "doc_id",
    fetch_multiplier: int = 3,   # parent dedupe 때문에 rerank 범위를 top_k보다 넓힘
) -> List[DocumentScore]:
    """
    child(청크)로 검색 + rerank + score 보존 → parent로 승격해서 반환

    parent 점수 = 해당 parent로 연결된 child들의 최소 distance score
    반환 순서 = rerank된 child 순서를 따라가되 parent 단위로 dedupe
    """

    # 1) child 후보 확보 (score 포함)
    candidates: List[Tuple[Document, float]] = vector_db.similarity_search_with_score(query, k=initial_k)
    if not candidates:
        return []

    # 2) rerank는 Document만 받음
    child_docs = [d for d, _ in candidates]

    # dedupe 때문에 top_k보다 넓게 rerank
    rerank_n = min(len(child_docs), max(top_k * fetch_multiplier, top_k))
    reranked_children = reranker.rerank(query=query, docs=child_docs, top_n=rerank_n)

    # 3) child score lookup
    score_map: Dict[str, float] = {}
    for d, s in candidates:
        # 기존 retrieval.py의 _doc_key 방식이 있으면 그걸 써도 됨
        src = (d.metadata or {}).get("source", "unknown")
        text = (d.page_content or "")
        k = f"{src}::{text[:200]}"
        score_map[k] = float(s)

    def _child_key(d: Document) -> str:
        src = (d.metadata or {}).get("source", "unknown")
        text = (d.page_content or "")
        return f"{src}::{text[:200]}"

    # 4) parent_id 기준으로 dedupe + parent_score(min child score)
    parent_best_score: Dict[str, float] = {}
    parent_first_child: Dict[str, Document] = {}  # parent metadata 보정용(예: source)
    parent_order: List[str] = []

    for child in reranked_children:
        pid = (child.metadata or {}).get(parent_id_key)
        if not pid:
            # parent id가 없으면 승격 불가 → skip(혹은 child를 그대로 쓰는 fallback도 가능)
            continue

        child_score = score_map.get(_child_key(child), 999.0)

        if pid not in parent_best_score:
            parent_best_score[pid] = child_score
            parent_first_child[pid] = child
            parent_order.append(pid)
        else:
            # 같은 parent에 더 좋은 child가 있으면 score 갱신
            if child_score < parent_best_score[pid]:
                parent_best_score[pid] = child_score

        if len(parent_order) >= top_k:
            # 이미 top_k개의 parent를 확보했으면 조기 종료 가능
            # (하지만 더 좋은 score를 찾고 싶으면 이 break를 제거해도 됨)
            pass

    if not parent_order:
        return []

    # 5) docstore에서 parent 문서 로드
    parent_docs: List[Optional[Document]] = docstore.mget(parent_order)

    # 6) 결과 조립 (parent 문서 + parent_score)
    results: List[DocumentScore] = []
    for pid, pdoc in zip(parent_order, parent_docs):
        if pdoc is None:
            continue

        # parent 문서에 source가 없으면 child의 source로 보정
        if (pdoc.metadata or {}).get("source") is None:
            child = parent_first_child.get(pid)
            if child is not None:
                pdoc.metadata = dict(pdoc.metadata or {})
                pdoc.metadata["source"] = (child.metadata or {}).get("source")

        results.append((pdoc, float(parent_best_score.get(pid, 999.0))))

        if len(results) >= top_k:
            break

    return results