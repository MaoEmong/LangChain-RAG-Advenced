"""
services/command_parser.py
============================================================
명령 JSON 파서

이 모듈은 LLM이 생성한 JSON 문자열을 파싱하고 검증합니다.
Pydantic을 사용하여 스키마 검증을 수행합니다.
"""

import json
from schemas.command import CommandResponse
from pydantic import ValidationError

def parse_command_json(text: str) -> CommandResponse | None:
    """
    LLM이 생성한 JSON 문자열을 CommandResponse 객체로 파싱/검증합니다.
    
    이 함수는:
    1. JSON 문자열을 Python 딕셔너리로 파싱
    2. Pydantic을 사용하여 스키마 검증
    3. CommandResponse 객체로 변환
    
    Args:
        text (str): LLM이 생성한 JSON 문자열
    
    Returns:
        CommandResponse | None: 파싱 성공 시 CommandResponse 객체,
                               실패 시 None
    
    Note:
        - JSON 형식 오류 시 None 반환
        - 스키마 검증 실패 시 None 반환
        - LLM이 잘못된 형식의 JSON을 생성할 수 있으므로 방어 코드 필요
    """
    try:
        # JSON 문자열을 Python 딕셔너리로 파싱
        data = json.loads(text)
        
        # Pydantic 모델로 검증 및 변환
        # - 필수 필드 확인
        # - 타입 검증
        # - 스키마 규칙 검증
        return CommandResponse.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        # JSON 파싱 오류 또는 스키마 검증 실패 시 None 반환
        return None
