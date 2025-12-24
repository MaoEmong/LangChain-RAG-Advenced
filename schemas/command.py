"""
schemas/command.py
============================================================
명령 스키마 정의

이 모듈은 LLM이 생성하는 명령 JSON의 스키마를 정의합니다.
Pydantic을 사용하여 타입 검증과 스키마 검증을 수행합니다.
"""

from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field

class CommandAction(BaseModel):
    """
    실행할 액션 하나를 나타내는 스키마
    
    Attributes:
        name (str): 실행할 함수 이름
        args (Dict[str, Any]): 함수에 전달할 인자 딕셔너리
    """
    # 실행할 함수 이름 (예: "OpenUrl", "ShowNotification")
    name: str = Field(..., description="실행할 함수 이름")

    # 함수 인자 딕셔너리 (인자가 없으면 빈 딕셔너리 {})
    args: Dict[str, Any] = Field(default_factory=dict)

class CommandResponse(BaseModel):
    """
    명령 응답 스키마
    
    LLM이 생성하는 명령 JSON의 전체 구조를 정의합니다.
    
    Attributes:
        type (Literal["command"]): 항상 "command"로 고정
        speech (str): 사용자에게 보여줄/말해줄 안내 문구
        actions (List[CommandAction]): 실행할 액션 목록
    """
    # 타입은 항상 "command"로 고정 (다른 타입과 구분)
    type: Literal["command"] = "command"

    # 사용자에게 보여줄/말해줄 한 문장
    # 예: "URL을 열었습니다", "알림을 표시했습니다"
    speech: str

    # 실행할 액션 목록 (여러 액션을 순차적으로 실행 가능)
    # 빈 배열이면 실행할 액션 없음 (설명/질문에 가까운 경우)
    actions: List[CommandAction] = Field(default_factory=list)
