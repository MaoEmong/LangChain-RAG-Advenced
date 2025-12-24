# schemas/intent.py
from typing import Literal
from pydantic import BaseModel

class IntentResult(BaseModel):
    intent: Literal["command","explain"]
    reason: str