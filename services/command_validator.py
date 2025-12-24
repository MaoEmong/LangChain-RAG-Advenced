# services/command_validator.py
from commands.registry import ALLOWED_COMMANDS
from schemas.command import CommandResponse

def validate_commands(cmd: CommandResponse) -> tuple[bool, str]:
    """
    - 명령 이름이 허용 목록에 있는지
    - 필요한 args가 있는지
    """
    for action in cmd.actions:
        if action.name not in ALLOWED_COMMANDS:
            return False, f"허용되지 않은 명령: {action.name}"

        expected_args = ALLOWED_COMMANDS[action.name]["args"]

        for arg in expected_args:
            if arg not in action.args:
                return False, f"명령 '{action.name}'에 필요한 인자 누락: {arg}"

    return True, "ok"
