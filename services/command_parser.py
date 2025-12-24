# services/command_parser.py
import json
from schemas.command import CommandResponse
from pydantic import ValidationError

def parse_command_json(text: str) -> CommandResponse | None:
    """
    LLM이 생성한 JSON 문자열을
    CommandResponse로 파싱/검증한다.
    실패하면 None 반환.
    """
    try:
        data = json.loads(text)
        return CommandResponse.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        return None
