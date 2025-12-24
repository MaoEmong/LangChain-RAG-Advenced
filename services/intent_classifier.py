"""
services/intent_classifier.py
============================================================
사용자 의도 분류기

이 모듈은 사용자의 입력을 분석하여 의도를 분류합니다:
- "command": 실행/조작 요청 (예: "음악 재생해줘")
- "explain": 설명/질문 요청 (예: "RAG가 뭐야?")

하이브리드 분류 방식:
1. Rule 기반: 빠른 패턴 매칭 (성능 최적화)
2. LLM 기반: 애매한 경우 LLM이 분류 (정확도 향상)
"""

import re
import json
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from schemas.intent import IntentResult
from prompts.intent_prompt import INTENT_PROMPT_TEMPLATE

# ============================================================
# Rule 기반 분류: 빠른 패턴 매칭
# ============================================================
# "행동"을 나타내는 대표적인 표현들 (정규식 패턴)
# 필요하면 계속 추가 가능
COMMAND_HINTS = [
    r"해줘", r"해주세요", r"해봐", r"해봐줘",
    r"켜줘", r"꺼줘",
    r"열어줘", r"닫아줘",
    r"재생해줘", r"틀어줘",
    r"저장해줘", r"복사해줘",
    r"이동해줘", r"바꿔줘", r"변경해줘",
    r"실행해줘", r"눌러줘", r"검색해줘",
]

# "설명/질문"을 나타내는 대표적인 표현들
EXPLAIN_HINTS = [
    r"뭐야", r"무슨", r"설명", r"원리", r"왜", r"어떻게",
    r"차이", r"정의", r"의미", r"개념",
]

def rule_intent(question: str) -> Optional[IntentResult]:
    """
    Rule 기반 의도 분류 (빠른 패턴 매칭)
    
    정규식 패턴을 사용하여 빠르게 의도를 분류합니다.
    명확한 패턴이 없으면 None을 반환하여 LLM 분류로 넘깁니다.
    
    Args:
        question (str): 사용자 입력 질문
    
    Returns:
        Optional[IntentResult]: 분류 결과 또는 None (애매한 경우)
    """
    q = question.strip()

    # 너무 짧은 입력은 안전하게 explain으로 처리
    if len(q) <= 2:
        return IntentResult(intent="explain", reason="too_short")

    # 명령 힌트 패턴 체크 (우선순위 높음)
    for pat in COMMAND_HINTS:
        if re.search(pat, q):
            return IntentResult(intent="command", reason=f"rule_match:{pat}")

    # 설명 힌트 패턴 체크
    for pat in EXPLAIN_HINTS:
        if re.search(pat, q):
            return IntentResult(intent="explain", reason=f"rule_match:{pat}")

    # 확신 없으면 None 반환 → LLM 분류로 넘김
    return None

# ============================================================
# LLM 기반 분류: 애매한 경우 정확한 분류
# ============================================================
def llm_intent(question: str, llm) -> IntentResult:
    """
    LLM 기반 의도 분류
    
    Rule 기반으로 분류가 안 될 때 LLM을 사용하여 분류합니다.
    LLM이 JSON 형식으로 의도와 근거를 반환합니다.
    
    Args:
        question (str): 사용자 입력 질문
        llm: 언어 모델 (ChatOpenAI 등)
    
    Returns:
        IntentResult: 분류 결과
    """
    # 프롬프트 템플릿 생성
    prompt = ChatPromptTemplate.from_template(INTENT_PROMPT_TEMPLATE)
    
    # LangChain 체인 구성: 프롬프트 -> LLM -> 문자열 파싱
    chain = prompt | llm | StrOutputParser()

    # LLM 호출하여 분류 결과 받기
    raw = chain.invoke({"question": question}).strip()

    # LLM이 JSON을 깔끔히 안 주는 경우 대비 (방어 코드)
    try:
        # JSON 파싱
        data = json.loads(raw)
        # IntentResult 객체로 변환
        return IntentResult(**data)
    except Exception:
        # 파싱 실패 시 안전하게 explain로 처리
        return IntentResult(intent="explain", reason="llm_parse_failed")

# ============================================================
# 메인 분류 함수: 하이브리드 방식
# ============================================================
def classify_intent(question: str, llm) -> IntentResult:
    """
    사용자 입력의 의도를 분류합니다 (하이브리드 방식)
    
    분류 전략:
    1. Rule 기반 분류 시도 (빠름, 비용 없음)
    2. Rule로 분류 안 되면 LLM 분류 (정확함, 비용 있음)
    
    Args:
        question (str): 사용자 입력 질문
        llm: 언어 모델 (ChatOpenAI 등)
    
    Returns:
        IntentResult: 분류 결과 (intent: "command" | "explain")
    """
    # 1단계: Rule 기반 분류 시도
    r = rule_intent(question)
    if r:
        return r  # 명확하게 분류되면 바로 반환

    # 2단계: 애매하면 LLM 분류
    return llm_intent(question, llm)