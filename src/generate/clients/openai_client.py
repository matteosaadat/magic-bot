# AI INSTRUCTION:
# Define a client for OpenAI Chat Completions API.
# It should follow the same interface as OllamaClient.

import os
from typing import List, Tuple, Dict, Any
from openai import OpenAI
from ..types import Message, ModelParams

class OpenAIClient:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def set_model(self, model: str):
        self.model = model

    def generate(self, messages: List[Message], params: ModelParams) -> Tuple[str, Dict[str, Any]]:
        formatted = [{"role": m.role, "content": m.content} for m in messages]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=formatted,
            temperature=params.temperature or 0.3,
            max_tokens=params.max_tokens or 1000,
        )
        text = resp.choices[0].message.content.strip()
        meta = {"engine": "openai", "model": self.model}
        return text, meta
