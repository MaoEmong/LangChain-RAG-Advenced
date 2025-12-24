"""
services/command_validator.py
============================================================
명령 검증기 (화이트리스트 기반)

이 모듈은 LLM이 생성한 명령이 안전한지 검증합니다.
화이트리스트 방식으로 허용된 명령만 실행 가능하도록 합니다.

검증 항목:
1. 명령 이름이 허용 목록에 있는지
2. 필요한 인자가 모두 있는지
"""

from commands.registry import ALLOWED_COMMANDS
from schemas.command import CommandResponse

def validate_commands(cmd: CommandResponse) -> tuple[bool, str]:
    """
    명령이 안전하고 실행 가능한지 검증합니다.
    
    검증 규칙:
    1. 각 액션의 이름이 ALLOWED_COMMANDS에 등록되어 있는지 확인
    2. 각 액션에 필요한 인자가 모두 포함되어 있는지 확인
    
    Args:
        cmd (CommandResponse): 검증할 명령 객체
    
    Returns:
        tuple[bool, str]: (검증 통과 여부, 이유 메시지)
            - (True, "ok"): 검증 통과
            - (False, "이유"): 검증 실패 및 실패 이유
    
    Note:
        - 화이트리스트 방식: 허용된 명령만 실행 가능
        - 블랙리스트가 아닌 화이트리스트를 사용하여 보안 강화
        - 새로운 명령 추가 시 commands/registry.py에 등록 필요
    """
    # 각 액션을 순회하며 검증
    for action in cmd.actions:
        # 1. 명령 이름이 허용 목록에 있는지 확인
        if action.name not in ALLOWED_COMMANDS:
            return False, f"허용되지 않은 명령: {action.name}"

        # 2. 필요한 인자 목록 가져오기
        expected_args = ALLOWED_COMMANDS[action.name]["args"]

        # 3. 필요한 인자가 모두 있는지 확인
        for arg in expected_args:
            if arg not in action.args:
                return False, f"명령 '{action.name}'에 필요한 인자 누락: {arg}"

    # 모든 검증 통과
    return True, "ok"
