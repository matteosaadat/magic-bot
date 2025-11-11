# AI INSTRUCTION:
# Define data models for search layer.
# These types represent what retrieval returns and how personas are structured.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class ContextChunk:
    """A small, retrievable text unit returned by the Retriever."""
    id: str
    text: str
    score: float
    source: str
    meta: Optional[Dict[str, Any]] = None


@dataclass
class Persona:
    """Describes a persona's tone, style, and behavior directives."""
    key: str
    name: str
    style: str
    directives: str
    meta: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Optional wrapper to hold results and metadata."""
    query: str
    chunks: List[ContextChunk]
    persona: Optional[Persona] = None
