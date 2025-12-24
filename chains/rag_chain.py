# chains/rag_chain.py
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ----------------------------
# Helper: docs -> context string
# ----------------------------
def format_docs(docs):
    """
    검색된 Document들을 LLM에 넣기 좋은 문자열로 변환한다.
    - 문서가 섞여 보이는 문제를 줄이기 위해
      각 문서마다 source(출처)를 붙여준다.
    """
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        blocks.append(f"[DOC {i}] source={src}\n{d.page_content}")
    return "\n\n".join(blocks)

def build_rag_chain(retriever, llm, prompt):
    """
    질문(문자열) -> 검색 -> 컨텍스트 구성 -> 답변 생성
    """
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
