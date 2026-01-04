"""
test_retrieval.py
============================================================
검색 기능 테스트 스크립트

Parent-Child 구조를 사용한 문서 검색 기능을 테스트합니다.

테스트 내용:
1. ChromaDB에서 Child 문서(청크) 검색
2. doc_id별로 문자열 매칭을 1순위로 랭킹
3. SQLiteDocStore에서 Parent 문서 복원

실행:
    python test_retrieval.py
"""

from collections import Counter, defaultdict

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from config import EMBED_MODEL, OPENAI_API_KEY, CHROMA_DIR
from docstore_sqlite import SQLiteDocStore
from config import DOCSTORE_PATH

COLLECTION_NAME = "my_rag_docs"


# =========================
# 1) DB 로더
# =========================
def build_or_load_chroma() -> Chroma:
    """
    ChromaDB 벡터 저장소를 생성하거나 로드합니다.
    
    Returns:
        Chroma: ChromaDB 벡터 저장소 객체
    """
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


# =========================
# 2) doc_id 랭킹 함수
# =========================
def rank_doc_ids_by_match_then_score(child_docs_with_score, query: str, top_n: int = 3):
    """
    doc_id를 기준으로 문서를 랭킹합니다.
    
    랭킹 규칙 (우선순위 순):
      1) match_count(desc): query 문자열이 실제로 포함된 child 수
      2) count(desc): top-k 결과에 얼마나 많이 섞였는지
      3) best_score(asc): doc_id 그룹 안에서 가장 좋은 score
    
    Args:
        child_docs_with_score: (Document, score) 튜플 리스트
        query: 검색 쿼리 문자열
        top_n: 반환할 상위 doc_id 개수
    
    Returns:
        tuple: (ranked_docids, docid_to_children, docid_match, docid_count, docid_best_score)
            - ranked_docids: 랭킹된 doc_id 리스트
            - docid_to_children: doc_id별 child 문서 매핑
            - docid_match: doc_id별 매칭 개수
            - docid_count: doc_id별 등장 개수
            - docid_best_score: doc_id별 최고 점수
    
    Note:
        - Chroma distance 계열은 낮을수록 좋으므로 best_score는 min()로 계산
        - 만약 score가 "높을수록 좋다"면 min->max, asc->desc로 변경 필요
    """
    docid_count = Counter()
    docid_match = Counter()
    docid_best_score = {}

    docid_to_children = defaultdict(list)

    for d, score in child_docs_with_score:
        doc_id = d.metadata.get("doc_id")
        if not doc_id:
            continue

        docid_count[doc_id] += 1
        docid_to_children[doc_id].append((d, score))

        text = d.page_content or ""
        if query in text:
            docid_match[doc_id] += 1

        if doc_id not in docid_best_score:
            docid_best_score[doc_id] = score
        else:
            docid_best_score[doc_id] = min(docid_best_score[doc_id], score)

    ranked_docids = sorted(
        docid_count.keys(),
        key=lambda doc_id: (
            -docid_match[doc_id],                 # match 우선
            -docid_count[doc_id],                 # 그 다음 count
            docid_best_score.get(doc_id, 999999), # 그 다음 best_score(낮을수록 좋음)
        )
    )[:top_n]

    return ranked_docids, docid_to_children, docid_match, docid_count, docid_best_score


# =========================
# 3) 메인
# =========================
def main():
    """
    검색 기능 테스트 메인 함수
    
    테스트 프로세스:
    1. ChromaDB와 DocStore 로드
    2. Child 문서 검색 (점수 포함)
    3. doc_id별 랭킹
    4. Parent 문서 복원
    5. 결과 출력
    """
    db = build_or_load_chroma()
    docstore = SQLiteDocStore(DOCSTORE_PATH)

    query = "APX3002345386815CN"  # 테스트용 쿼리 (여기서 변경 가능)

    # 점수 포함 검색 (k는 넉넉하게 설정)
    # similarity_search_with_score: [(Document, score), ...] 형태로 반환
    child_docs_with_score = db.similarity_search_with_score(query, k=50)

    # Child 검색 결과 출력
    print("==== CHILD RESULTS (from Chroma) ====")
    print("count:", len(child_docs_with_score))

    # Child 문서 출력 (상위 일부)
    for i, (d, score) in enumerate(child_docs_with_score[:20], 1):
        src = d.metadata.get("source")
        page = d.metadata.get("page")
        doc_id = d.metadata.get("doc_id")
        print(f"\n[CHILD {i}] score={score} source={src} page={page} doc_id={doc_id}")
        print((d.page_content or "")[:350])

    # doc_id 랭킹 (match -> count -> score 순서)
    ranked_docids, docid_to_children, docid_match, docid_count, docid_best_score = (
        rank_doc_ids_by_match_then_score(child_docs_with_score, query, top_n=3)
    )

    # doc_id 랭킹 디버그 정보 출력
    print("\n=== DOCID RANK DEBUG (match -> count -> best_score) ===")
    for idx, doc_id in enumerate(ranked_docids, 1):
        print(
            f"[{idx}] doc_id={doc_id} | match={docid_match[doc_id]} | "
            f"count={docid_count[doc_id]} | best_score={docid_best_score.get(doc_id)}"
        )

    # Parent 문서 복원 (랭킹된 doc_id만 사용)
    parents = []
    parent_list = docstore.mget(ranked_docids)
    for p in parent_list:
        if p:
            parents.append(p)

    # Parent 복원 결과 출력
    print("\n==== PARENT RESTORED (top doc_id groups) ====")
    print("count:", len(parents))

    for i, p in enumerate(parents, 1):
        print(
            f"\n[PARENT {i}] source={p.metadata.get('source')} doc_id={p.metadata.get('doc_id')}"
        )
        print((p.page_content or "")[:1200])


if __name__ == "__main__":
    main()
