"""
schemas/intent.py
============================================================
의도 분류 결과 스키마

이 모듈은 사용자 입력의 의도 분류 결과를 나타내는 스키마를 정의합니다.
"""

from typing import Literal
from pydantic import BaseModel

class IntentResult(BaseModel):
    """
    의도 분류 결과 스키마
    
    사용자 입력을 분석한 결과를 나타냅니다.
    
    Attributes:
        intent (Literal["command", "explain"]): 분류된 의도
            - "command": 실행/조작 요청
            - "explain": 설명/질문 요청
        reason (str): 분류 근거 (디버깅/로깅용)
    """
    # 분류된 의도: "command" 또는 "explain"
    intent: Literal["command", "explain"]
    
    # 분류 근거 (예: "rule_match:해줘", "llm_parse_failed")
    reason: str