"""RouterBackend — fan-out writes and merged reads across multiple backends.

Realizes the "long memory in different frameworks at once" goal. A RouterBackend
*is* a LongTermBackend (same Protocol), so MemoryManager treats it like any other
single backend — it just happens to wrap several named children.

Write: fan out to all children (or to one named child via metadata["backend"], or
       to only the first when write="first"). One child failing doesn't lose the
       write to the others; if every target fails, the error propagates.
Read:  query each child, then merge. Scores are NOT comparable across
       heterogeneous backends (cosine vs mem0 score vs a graph blob), so the
       default merge is "interleave" (round-robin, balanced representation);
       "score" is available when you know scores are comparable. Results are
       deduped by normalized text and tagged with metadata["backend"].

Selected via LONG_TERM_BACKEND="vector+mem0" (a "+"/"," separated list).
"""

from __future__ import annotations

from .base import LongTermBackend
from ..types import MemoryItem, Message


class RouterBackend:
    def __init__(
        self,
        backends: dict[str, LongTermBackend],
        merge: str = "interleave",
        write: str = "all",
    ):
        if not backends:
            raise ValueError("RouterBackend needs at least one child backend")
        self.backends = backends
        self.merge = merge
        self.write = write
        self.name = "router(" + "+".join(backends) + ")"
        self.errors: list[tuple[str, Exception]] = []

    # -- write ---------------------------------------------------------------
    def _write_targets(self, metadata: dict | None) -> list[str]:
        # explicit per-call routing wins: metadata={"backend": "mem0"}
        if metadata and metadata.get("backend") in self.backends:
            return [metadata["backend"]]
        if self.write == "first":
            return [next(iter(self.backends))]
        return list(self.backends)

    def add(
        self, messages: list[Message], user_id: str, metadata: dict | None = None
    ) -> list[MemoryItem]:
        targets = self._write_targets(metadata)
        out: list[MemoryItem] = []
        failures: list[Exception] = []
        for name in targets:
            try:
                items = self.backends[name].add(messages, user_id, metadata)
            except Exception as e:  # resilience: keep writing to other backends
                self.errors.append((name, e))
                failures.append(e)
                continue
            for it in items:
                it.metadata = {**it.metadata, "backend": name}
                out.append(it)
        if failures and len(failures) == len(targets):
            raise failures[-1]  # every target failed — surface it
        return out

    # -- read ----------------------------------------------------------------
    def search(self, query: str, user_id: str, limit: int = 5) -> list[MemoryItem]:
        per_backend: dict[str, list[MemoryItem]] = {}
        for name, backend in self.backends.items():
            try:
                items = backend.search(query, user_id, limit)
            except Exception as e:
                self.errors.append((name, e))
                continue
            for it in items:
                it.metadata = {**it.metadata, "backend": name}
            per_backend[name] = items
        return (
            self._merge_score(per_backend, limit)
            if self.merge == "score"
            else self._merge_interleave(per_backend, limit)
        )

    @staticmethod
    def _dedup_key(item: MemoryItem) -> str:
        return item.text.strip().lower()

    def _merge_interleave(
        self, per_backend: dict[str, list[MemoryItem]], limit: int
    ) -> list[MemoryItem]:
        pools = list(per_backend.values())
        pos = [0] * len(pools)
        seen: set[str] = set()
        out: list[MemoryItem] = []
        while len(out) < limit and any(pos[i] < len(pools[i]) for i in range(len(pools))):
            for i in range(len(pools)):
                if pos[i] >= len(pools[i]):
                    continue
                it = pools[i][pos[i]]
                pos[i] += 1
                key = self._dedup_key(it)
                if key not in seen:
                    seen.add(key)
                    out.append(it)
                    if len(out) >= limit:
                        break
        return out

    def _merge_score(
        self, per_backend: dict[str, list[MemoryItem]], limit: int
    ) -> list[MemoryItem]:
        items = [it for lst in per_backend.values() for it in lst]
        items.sort(key=lambda m: m.score, reverse=True)
        seen: set[str] = set()
        out: list[MemoryItem] = []
        for it in items:
            key = self._dedup_key(it)
            if key not in seen:
                seen.add(key)
                out.append(it)
                if len(out) >= limit:
                    break
        return out

    # -- get_all / delete ----------------------------------------------------
    def get_all(self, user_id: str) -> list[MemoryItem]:
        seen: set[str] = set()
        out: list[MemoryItem] = []
        for name, backend in self.backends.items():
            try:
                items = backend.get_all(user_id)
            except NotImplementedError:
                continue  # e.g. lightrag has no flat list
            except Exception as e:
                self.errors.append((name, e))
                continue
            for it in items:
                it.metadata = {**it.metadata, "backend": name}
                key = self._dedup_key(it)
                if key not in seen:
                    seen.add(key)
                    out.append(it)
        return out

    def delete(self, memory_id: str, user_id: str) -> None:
        # ids are namespaced per backend; we don't know the owner, so try each
        # and ignore backends that don't recognize it.
        for name, backend in self.backends.items():
            try:
                backend.delete(memory_id, user_id)
            except (NotImplementedError, KeyError, ValueError):
                continue
            except Exception as e:
                self.errors.append((name, e))
