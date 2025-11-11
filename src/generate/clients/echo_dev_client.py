# AI INSTRUCTION:
# Provide a dummy model client for local dev and testing without API calls.

from typing import List, Tuple, Dict, Any
from ..types import Message, ModelParams

class EchoDevClient:
    def __init__(self):
        self.model = "echo-dev"

    def generate(self, messages: List[Message], params: ModelParams) -> Tuple[str, Dict[str, Any]]:
        user_inputs = [m.content for m in messages if m.role == "user"]
        text = f"[ECHO RESPONSE]\n{user_inputs[-1] if user_inputs else '(no user input)'}"
        meta = {"engine": "echo", "model": "echo-dev", "temp": params.temperature, "max_tokens": params.max_tokens}
        return text, meta
