# chains/command_chain.py
# ------------------------------------------------------------
# Command Chain
# - question -> (retrieve) -> context -> prompt -> llm -> json text
# ------------------------------------------------------------

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from chains.rag_chain import format_docs  # 이미 만든 함수 재사용

def build_command_chain(retriever, llm, prompt):
    """
    Command chain: 사용자 입력 -> 문서 검색 -> JSON 명령 생성
    반환값은 "문자열(JSON)" 이다.
    """
    return (
        {
            "context": retriever,        # retriever가 docs를 반환
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
