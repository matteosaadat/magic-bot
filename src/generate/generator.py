# AI INSTRUCTION:
# Provide a ChatGenerator class that:
# - accepts any model client (Ollama, OpenAI, Echo)
# - builds prompts from persona + context
# - returns ChatResponse
# - uses generate/types.py for data structure consistency

from __future__ import annotations
import yaml
import os
from typing import List, Optional
from .types import Message, ChatResponse, ModelParams
from src.search.types import ContextChunk, Persona
from src.search.prompts import build_system_prompt


class ChatGenerator:
    def __init__(self, model_client, config_path: str = "src/generate/config.yaml", persona_path: Optional[str] = None):
        self.model_client = model_client
        self.config_path = config_path
        self.persona_path = persona_path
        self.cfg = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _compose_system_message(self, persona: Optional[Persona]) -> str:
        """Build system message combining persona + global rules."""
        if persona:
            return build_system_prompt(persona.name, persona.style, persona.directives)
        base_prompt = self.cfg.get("system_prompt", "")
        return base_prompt.strip()

    def _compose_user_message(self, query: str, context: List[ContextChunk]) -> str:
        """Attach context snippets to the user query."""
        ctx_txt = "\n\n---\n\n".join(
            [f"[{c.id}] (score={c.score:.3f}) source={c.source}\n{c.text}" for c in context]
        )
        return f"User question:\n{query}\n\nContext:\n{ctx_txt}"

    def chat(
        self,
        user_message: str,
        history: List[Message],
        context: List[ContextChunk],
        persona: Optional[Persona] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        persona_key: Optional[str] = None,  # backward compatibility
    ) -> ChatResponse:
        """Main entry point for generation."""
        sys_msg = self._compose_system_message(persona)
        user_msg = self._compose_user_message(user_message, context)
        messages = [Message(role="system", content=sys_msg), *history, Message(role="user", content=user_msg)]

        params = ModelParams(
            temperature=temperature or self.cfg.get("temperature", 0.3),
            max_tokens=max_tokens or self.cfg.get("max_tokens", 1000),
        )

        response_text, meta = self.model_client.generate(messages, params)
        citations = [c.id for c in context]
        return ChatResponse(text=response_text, citations=citations, meta=meta)
