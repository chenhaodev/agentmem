"""Short-term memory: a lightweight, framework-agnostic, shared recent-turn buffer.

Deliberately NOT backed by mem0/Letta — those add LLM-extraction latency that
has no place on the hot path. This is just a per-session ring buffer in a shared
namespace, so multiple agents/handlers see the same recent context.

Two interchangeable implementations behind one `ShortTermMemory` Protocol:
  - InProcessShortTermMemory : shared within one process (zero deps).
  - RedisShortTermMemory     : shared across processes / machines ("quickly
                               shared" at scale), selected with SHORT_TERM_STORE=redis.
Switching is a config change; nothing else in the system cares which is used.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Deque, Protocol, runtime_checkable

from .config import Config
from .types import Message


@runtime_checkable
class ShortTermMemory(Protocol):
    def append(self, session_id: str, message: Message) -> None: ...
    def recent(self, session_id: str, limit: int | None = None) -> list[Message]: ...
    def clear(self, session_id: str) -> None: ...
    def sessions(self) -> list[str]: ...


class InProcessShortTermMemory:
    def __init__(self, max_turns: int = 12):
        self.max_turns = max_turns
        self._buffers: dict[str, Deque[Message]] = defaultdict(
            lambda: deque(maxlen=max_turns)
        )

    def append(self, session_id: str, message: Message) -> None:
        self._buffers[session_id].append(message)

    def recent(self, session_id: str, limit: int | None = None) -> list[Message]:
        msgs = list(self._buffers[session_id])
        return msgs[-limit:] if limit else msgs

    def clear(self, session_id: str) -> None:
        self._buffers.pop(session_id, None)

    def sessions(self) -> list[str]:
        return list(self._buffers.keys())


class RedisShortTermMemory:
    """Cross-process short-term buffer using a Redis list per session.

    Each session is a capped list (LPUSH + LTRIM) under
    `{namespace}:{session_id}`, so any process pointed at the same Redis sees the
    same recent turns — this is the "quickly shared" path at scale.

    Requires: pip install redis  (and a reachable Redis server).
    """

    def __init__(self, config: Config):
        try:
            import redis
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "SHORT_TERM_STORE=redis but redis not installed. Run: pip install redis"
            ) from e
        self.max_turns = config.short_term_max_turns
        self.ns = config.redis_namespace
        self._r = redis.Redis.from_url(config.redis_url, decode_responses=True)

    def _key(self, session_id: str) -> str:
        return f"{self.ns}:{session_id}"

    def append(self, session_id: str, message: Message) -> None:
        key = self._key(session_id)
        pipe = self._r.pipeline()
        pipe.rpush(key, json.dumps(message.as_dict()))
        pipe.ltrim(key, -self.max_turns, -1)  # keep only the last N
        pipe.execute()

    def recent(self, session_id: str, limit: int | None = None) -> list[Message]:
        raw = self._r.lrange(self._key(session_id), 0, -1)
        msgs = [Message(**json.loads(r)) for r in raw]
        return msgs[-limit:] if limit else msgs

    def clear(self, session_id: str) -> None:
        self._r.delete(self._key(session_id))

    def sessions(self) -> list[str]:
        prefix = f"{self.ns}:"
        return [k[len(prefix):] for k in self._r.scan_iter(match=f"{prefix}*")]


def build_short_term(config: Config) -> ShortTermMemory:
    """Select the short-term store from config (SHORT_TERM_STORE)."""
    store = config.short_term_store.lower()
    if store == "memory":
        return InProcessShortTermMemory(config.short_term_max_turns)
    if store == "redis":
        return RedisShortTermMemory(config)
    raise ValueError(
        f"Unknown SHORT_TERM_STORE={store!r}. Expected one of: memory, redis."
    )
