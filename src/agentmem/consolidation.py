"""Background consolidation worker.

Moves the slow part of memory consolidation — LLM fact extraction / graph
building inside a long-term backend.add() — OFF the add_turn hot path. A single
daemon thread drains a job queue so the user's write call returns immediately.

Why ONE worker (not a pool): the long-term backends are not thread-safe — the
vector backend mutates a plain dict, mem0 drives an embedded Qdrant, and the
LightRAG adapter runs its own asyncio loop. Serializing every backend call
through one thread (guarded by a shared lock that readers also take) avoids
concurrent-mutation races without making each backend thread-safe.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable

from .types import Message

# what the worker does with a drained job: persist these turns for this user
ProcessFn = Callable[[list[Message], str], object]


class ConsolidationWorker:
    def __init__(self, process: ProcessFn, backend_lock: threading.RLock):
        self._process = process
        self._lock = backend_lock  # shared with the manager's reads
        self._q: "queue.Queue" = queue.Queue()
        self._errors: list[Exception] = []
        self._thread = threading.Thread(
            target=self._run, name="agentmem-consolidation", daemon=True
        )
        self._started = False

    def _ensure_started(self) -> None:
        if not self._started:
            self._thread.start()
            self._started = True

    def _run(self) -> None:
        while True:
            job = self._q.get()
            try:
                if job is None:  # shutdown sentinel
                    return
                messages, user_id = job
                with self._lock:
                    self._process(messages, user_id)
            except Exception as e:  # never let the worker thread die silently
                self._errors.append(e)
            finally:
                self._q.task_done()

    def submit(self, messages: list[Message], user_id: str) -> None:
        self._ensure_started()
        self._q.put((messages, user_id))

    def flush(self) -> None:
        """Block until every submitted job has been processed."""
        if self._started:
            self._q.join()

    @property
    def errors(self) -> list[Exception]:
        return list(self._errors)

    def shutdown(self) -> None:
        if self._started:
            self._q.put(None)
            self._thread.join(timeout=5)
            self._started = False
