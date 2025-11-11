# AI INSTRUCTION:
# Define a client for Ollama local inference.
# It must accept model name and expose generate(messages, params).

import requests
import os
from typing import List, Tuple, Dict, Any
from ..types import Message, ModelParams

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

class OllamaClient:
    def __init__(self, model: str = "mistral:7b-instruct"):
        self.model = model

    def set_model(self, model: str):
        self.model = model

    def generate(self, messages: List[Message], params: ModelParams) -> Tuple[str, Dict[str, Any]]:
        prompt = self._compose_prompt(messages)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(params.temperature or 0.3),
                "num_predict": int(params.max_tokens or 1000),
            },
        }
        url = f"{OLLAMA_HOST}/api/generate"
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip(), {"engine": "ollama", "model": self.model}

    def _compose_prompt(self, messages: List[Message]) -> str:
        parts = []
        for m in messages:
            parts.append(f"{m.role.upper()}:\n{m.content.strip()}\n")
        return "\n".join(parts)
