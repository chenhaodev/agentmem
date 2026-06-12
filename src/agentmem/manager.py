"""MemoryManager — the middleware facade tying short-term and long-term together.

Write path:  add_turn() -> short-term buffer (always) -> consolidate to long-term
             every N user turns. Consolidation (LLM extraction) runs either
             inline (default) or on a background worker when consolidation_async
             is set, so the add_turn hot path never blocks on the LLM.
Read path:   build_context() -> recent raw turns (short-term)
                               + relevance-ranked memories (long-term).

The backend behind long-term is whatever LONG_TERM_BACKEND selects; this class
never imports a specific framework, so swapping mem0<->vector<->lightrag changes
nothing here.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .backends import LongTermBackend, build_backend
from .config import Config
from .consolidation import ConsolidationWorker
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
        self._pending_lock = threading.Lock()
        # serializes ALL long_term access (worker writes + main-thread reads),
        # so the non-thread-safe backends stay consistent under async.
        self._backend_lock = threading.RLock()
        self._worker: ConsolidationWorker | None = (
            ConsolidationWorker(self._persist, self._backend_lock)
            if self.config.consolidation_async
            else None
        )

    # -- guarded long-term access --------------------------------------------
    def _persist(self, messages: list[Message], user_id: str) -> list[MemoryItem]:
        with self._backend_lock:
            return self.long_term.add(messages, user_id=user_id)

    def _search(self, query: str, user_id: str, limit: int) -> list[MemoryItem]:
        with self._backend_lock:
            return self.long_term.search(query, user_id=user_id, limit=limit)

    # -- write path -----------------------------------------------------------
    def add_turn(self, session_id: str, user_id: str, message: Message) -> None:
        with self._pending_lock:
            self.short_term.append(session_id, message)
            bucket = self._pending.setdefault(session_id, [])
            bucket.append(message)
            ready: list[Message] | None = None
            if message.role == "user":
                user_turns = sum(1 for m in bucket if m.role == "user")
                if user_turns >= self.config.consolidate_every:
                    ready, self._pending[session_id] = bucket, []
        if ready:
            if self._worker is not None:
                self._worker.submit(ready, user_id)  # off the hot path
            else:
                self._persist(ready, user_id)

    def consolidate(self, session_id: str, user_id: str) -> list[MemoryItem]:
        """Synchronously promote not-yet-persisted turns into long-term memory."""
        with self._pending_lock:
            pending = self._pending.get(session_id) or []
            self._pending[session_id] = []
        if not pending:
            return []
        return self._persist(pending, user_id)

    def flush(self) -> None:
        """Wait for any in-flight background consolidation to finish.

        No-op in synchronous mode. Call before recall() if you need memories from
        a just-submitted async consolidation to be searchable.
        """
        if self._worker is not None:
            self._worker.flush()

    # -- read path ------------------------------------------------------------
    def build_context(
        self, session_id: str, user_id: str, query: str, k: int = 5
    ) -> MemoryContext:
        return MemoryContext(
            recent=self.short_term.recent(session_id),
            memories=self._search(query, user_id, k),
        )

    # -- convenience ----------------------------------------------------------
    def remember(self, user_id: str, text: str) -> list[MemoryItem]:
        """Directly store a fact in long-term memory (no session)."""
        return self._persist([Message("user", text)], user_id)

    def recall(self, user_id: str, query: str, k: int = 5) -> list[MemoryItem]:
        return self._search(query, user_id, k)

    def end_session(self, session_id: str, user_id: str) -> list[MemoryItem]:
        self.flush()  # drain async jobs already submitted for this session
        stored = self.consolidate(session_id, user_id)
        self.short_term.clear(session_id)
        with self._pending_lock:
            self._pending.pop(session_id, None)
        return stored

    @property
    def consolidation_errors(self) -> list[Exception]:
        """Exceptions raised by background consolidation jobs (async mode)."""
        return self._worker.errors if self._worker is not None else []

    def close(self) -> None:
        """Stop the background worker (if any). Safe to call multiple times."""
        if self._worker is not None:
            self._worker.shutdown()

    def __enter__(self) -> "MemoryManager":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
