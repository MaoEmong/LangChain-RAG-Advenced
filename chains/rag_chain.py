"""
chains/rag_chain.py
============================================================
RAG (Retrieval-Augmented Generation) 체인 구성

이 모듈은 LangChain을 사용하여 RAG 파이프라인을 구성합니다:
1. 사용자 질문 입력
2. 벡터DB에서 관련 문서 검색
3. 검색된 문서를 컨텍스트 문자열로 변환
4. 프롬프트에 컨텍스트와 질문 삽입
5. LLM이 답변 생성
"""

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# Helper 함수: Document 리스트를 컨텍스트 문자열로 변환
# ============================================================
def format_docs(docs):
    """
    검색된 Document들을 LLM에 넣기 좋은 문자열로 변환합니다.
    
    각 문서에 번호와 출처를 붙여서:
    - 문서들이 구분되도록 함
    - 답변 생성 시 출처 추적 가능
    - 문서가 섞여 보이는 문제 방지
    
    Args:
        docs: 검색된 Document 객체 리스트
    
    Returns:
        str: 포맷팅된 컨텍스트 문자열
    """
    blocks = []
    for i, d in enumerate(docs, start=1):
        # 문서 출처 정보 (파일 경로 등)
        src = d.metadata.get("source", "unknown")
        # [DOC 1] source=경로\n문서내용 형식으로 구성
        blocks.append(f"[DOC {i}] source={src}\n{d.page_content}")
    
    # 문서들을 빈 줄 두 개로 구분하여 결합
    return "\n\n".join(blocks)

# ============================================================
# RAG 체인 빌더
# ============================================================
def build_rag_chain(retriever, llm, prompt):
    """
    RAG 체인을 구성하고 반환합니다.
    
    파이프라인 흐름:
    1. retriever: 질문으로 관련 문서 검색
    2. format_docs: 검색된 문서를 컨텍스트 문자열로 변환
    3. prompt: 컨텍스트와 질문을 프롬프트에 삽입
    4. llm: LLM이 답변 생성
    5. StrOutputParser: LLM 출력을 문자열로 파싱
    
    Args:
        retriever: 벡터DB 검색기 (LangChain Retriever)
        llm: 언어 모델 (ChatOpenAI 등)
        prompt: 프롬프트 템플릿 (ChatPromptTemplate)
    
    Returns:
        Runnable: LangChain Runnable 체인 객체
    """
    return (
        {
            # "context" 키: retriever로 문서 검색 후 format_docs로 변환
            "context": retriever | format_docs,
            # "question" 키: 입력 질문을 그대로 전달
            "question": RunnablePassthrough(),
        }
        | prompt      # 프롬프트에 context와 question 삽입
        | llm         # LLM이 답변 생성
        | StrOutputParser()  # LLM 출력을 문자열로 변환
    )
