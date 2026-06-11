"""MemoryManager — the middleware facade tying short-term and long-term together.

Write path:  add_turn() -> short-term buffer (always) -> consolidate to long-term
             every N user turns (LLM extraction happens here, off the hot path).
Read path:   build_context() -> recent raw turns (short-term)
                               + relevance-ranked memories (long-term).

The backend behind long-term is whatever LONG_TERM_BACKEND selects; this class
never imports a specific framework, so swapping mem0<->vector<->letta changes
nothing here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .backends import LongTermBackend, build_backend
from .config import Config
from .llm import DeepSeekLLM
from .short_term import ShortTermMemory, build_short_term
from .types import MemoryItem, Message


@dataclass
class MemoryContext:
    """What you feed into your next LLM prompt."""

    recent: list[Message]          # short-term: verbatim recent turns
    memories: list[MemoryItem]     # long-term: retrieved relevant facts

    def as_prompt_block(self) -> str:
        lines = []
        if self.memories:
            lines.append("# Relevant long-term memory")
            lines += [f"- {m.text}" for m in self.memories]
        if self.recent:
            lines.append("\n# Recent conversation")
            lines += [f"{m.role}: {m.content}" for m in self.recent]
        return "\n".join(lines)


class MemoryManager:
    def __init__(self, config: Config | None = None):
        self.config = config or Config.from_env()
        self.llm = DeepSeekLLM(self.config)
        self.short_term: ShortTermMemory = build_short_term(self.config)
        self.long_term: LongTermBackend = build_backend(self.config, self.llm)
        # turns added but not yet promoted to long-term, per session. Kept
        # separate from the short-term recency window so consolidation never
        # re-persists the same turns (which would duplicate memories).
        self._pending: dict[str, list[Message]] = {}

    # -- write path -----------------------------------------------------------
    def add_turn(self, session_id: str, user_id: str, message: Message) -> None:
        self.short_term.append(session_id, message)
        self._pending.setdefault(session_id, []).append(message)
        if message.role == "user":
            user_turns = sum(1 for m in self._pending[session_id] if m.role == "user")
            if user_turns >= self.config.consolidate_every:
                self.consolidate(session_id, user_id)

    def consolidate(self, session_id: str, user_id: str) -> list[MemoryItem]:
        """Promote not-yet-persisted turns into long-term memory."""
        pending = self._pending.get(session_id) or []
        if not pending:
            return []
        stored = self.long_term.add(pending, user_id=user_id)
        self._pending[session_id] = []
        return stored

    # -- read path ------------------------------------------------------------
    def build_context(
        self, session_id: str, user_id: str, query: str, k: int = 5
    ) -> MemoryContext:
        return MemoryContext(
            recent=self.short_term.recent(session_id),
            memories=self.long_term.search(query, user_id=user_id, limit=k),
        )

    # -- convenience ----------------------------------------------------------
    def remember(self, user_id: str, text: str) -> list[MemoryItem]:
        """Directly store a fact in long-term memory (no session)."""
        return self.long_term.add([Message("user", text)], user_id=user_id)

    def recall(self, user_id: str, query: str, k: int = 5) -> list[MemoryItem]:
        return self.long_term.search(query, user_id=user_id, limit=k)

    def end_session(self, session_id: str, user_id: str) -> list[MemoryItem]:
        stored = self.consolidate(session_id, user_id)
        self.short_term.clear(session_id)
        self._pending.pop(session_id, None)
        return stored
