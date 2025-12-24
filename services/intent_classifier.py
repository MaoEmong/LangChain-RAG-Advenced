# services/intent_classifier.py
import re
import json
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from schemas.intent import IntentResult
from prompts.intent_prompt import INTENT_PROMPT_TEMPLATE


# ----------------------------
# 1) 빠른 Rule 기반 분류
# ----------------------------
# "행동"을 나타내는 대표적인 표현들(필요하면 계속 추가)
COMMAND_HINTS = [
    r"해줘", r"해주세요", r"해봐", r"해봐줘",
    r"켜줘", r"꺼줘",
    r"열어줘", r"닫아줘",
    r"재생해줘", r"틀어줘",
    r"저장해줘", r"복사해줘",
    r"이동해줘", r"바꿔줘", r"변경해줘",
    r"실행해줘", r"눌러줘", r"검색해줘",
]

EXPLAIN_HINTS = [
    r"뭐야", r"무슨", r"설명", r"원리", r"왜", r"어떻게",
    r"차이", r"정의", r"의미", r"개념",
]

def rule_intent(question: str) -> Optional[IntentResult]:
    q = question.strip()

    # 너무 짧으면 explain으로 두는 게 안전
    if len(q) <= 2:
        return IntentResult(intent="explain", reason="too_short")

    # 명령 힌트 먼저 체크
    for pat in COMMAND_HINTS:
        if re.search(pat, q):
            return IntentResult(intent="command", reason=f"rule_match:{pat}")

    # 설명 힌트 체크
    for pat in EXPLAIN_HINTS:
        if re.search(pat, q):
            return IntentResult(intent="explain", reason=f"rule_match:{pat}")

    # 확신 없으면 None → LLM로 넘김
    return None


# ----------------------------
# 2) 애매하면 LLM 분류
# ----------------------------
def llm_intent(question: str, llm) -> IntentResult:
    prompt = ChatPromptTemplate.from_template(INTENT_PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    raw = chain.invoke({"question": question}).strip()

    # LLM이 JSON을 깔끔히 안 주는 경우 대비(최소 방어)
    try:
        data = json.loads(raw)
        return IntentResult(**data)
    except Exception:
        # 안전하게 explain로 처리
        return IntentResult(intent="explain", reason="llm_parse_failed")


def classify_intent(question: str, llm) -> IntentResult:
    # 1) Rule 우선
    r = rule_intent(question)
    if r:
        return r

    # 2) 애매하면 LLM
    return llm_intent(question, llm)