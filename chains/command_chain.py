"""
chains/command_chain.py
============================================================
Command 생성 체인 구성

이 모듈은 자연어 명령을 JSON 형식의 실행 가능한 명령으로 변환하는
LangChain 체인을 구성합니다.

파이프라인 흐름:
1. 사용자 입력 (자연어 명령)
2. 벡터DB에서 관련 함수/명령 정보 검색
3. 프롬프트에 컨텍스트와 사용자 입력 삽입
4. LLM이 JSON 형식의 명령 생성
5. JSON 문자열 반환
"""

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 참고: command_chain은 format_docs를 사용하지 않음
# 프롬프트에서 직접 Document 리스트를 처리하도록 설계됨

def build_command_chain(retriever, llm, prompt):
    """
    Command 생성 체인을 구성하고 반환합니다.
    
    사용자의 자연어 명령을 분석하여:
    - 문서에서 관련 함수 정보 검색
    - LLM이 JSON 형식의 명령 생성
    
    파이프라인 흐름:
    1. retriever: 질문으로 관련 문서 검색 (Document 리스트 반환)
    2. prompt: 컨텍스트(Document 리스트)와 질문을 프롬프트에 삽입
    3. llm: LLM이 JSON 형식의 명령 생성
    4. StrOutputParser: LLM 출력을 문자열로 파싱
    
    Args:
        retriever: 벡터DB 검색기 (LangChain Retriever)
        llm: 언어 모델 (ChatOpenAI 등)
        prompt: 명령 생성용 프롬프트 템플릿 (ChatPromptTemplate)
    
    Returns:
        Runnable: LangChain Runnable 체인 객체
        반환값은 JSON 문자열입니다.
    """
    return (
        {
            # "context" 키: retriever가 Document 리스트를 직접 반환
            # (프롬프트에서 직접 처리하도록 설계)
            "context": retriever,
            # "question" 키: 입력 질문을 그대로 전달
            "question": RunnablePassthrough(),
        }
        | prompt      # 프롬프트에 context와 question 삽입
        | llm         # LLM이 JSON 명령 생성
        | StrOutputParser()  # LLM 출력을 문자열로 변환
    )
