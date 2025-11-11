# AI INSTRUCTION:
# Define simple, typed dataclasses shared across generator modules.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class Message:
    """Single chat turn: system, user, or assistant."""
    role: str
    content: str


@dataclass
class ModelParams:
    """LLM parameters per request."""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass
class ChatResponse:
    """Final response from the generator."""
    text: str
    citations: List[str]
    meta: Dict[str, Any]
