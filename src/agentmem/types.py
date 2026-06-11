"""Shared data types used across short-term and long-term memory layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """A single conversational turn."""

    role: str  # "user" | "assistant" | "system"
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class MemoryItem:
    """A long-term memory record, normalized across every backend.

    This is the lowest-common-denominator shape. Backend-specific extras
    (e.g. Zep temporal edges, Letta block names) go in `metadata` so the
    common interface stays portable.
    """

    id: str
    text: str
    score: float = 0.0  # relevance for a search result; 0.0 when not from search
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # nicer demo output
        s = f"{self.score:.3f}" if self.score else "—"
        return f"MemoryItem(score={s}, text={self.text!r})"
