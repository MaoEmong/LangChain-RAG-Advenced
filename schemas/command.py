# schemas/command.py
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field

class CommandAction(BaseModel):
    # 실행할 함수 이름
    name: str = Field(..., description="실행할 함수 이름")

    # 함수 인자(없으면 {})
    args: Dict[str, Any] = Field(default_factory=dict)

class CommandResponse(BaseModel):
    # type은 항상 "command"로 고정
    type: Literal["command"] = "command"

    # 사용자에게 보여줄/읽어줄 한 문장
    speech: str

    # 실행 후보 액션 목록
    actions: List[CommandAction] = Field(default_factory=list)
