"""Tracing event dataclasses and TracingStore for memory visualization."""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from typemem.store import MemoryStore
from typemem.types import MemoryEntry, SearchResult


@dataclass
class StoreEvent:
    """A single store operation."""

    operation: str
    details: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConsolidationEvent:
    """A consolidation run with inputs/outputs."""

    name: str
    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    deletions: list[str]
    duration_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class InjectionEvent:
    """An injection query with results."""

    name: str
    query: str
    token_budget: int
    search_results: list[dict[str, Any]]
    context: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)


class TracingStore(MemoryStore):
    """MemoryStore wrapper that records all operations as events."""

    def __init__(self, inner: MemoryStore, max_events: int = 10_000):
        self.inner = inner
        self._max_events = max_events
        self._events: deque[StoreEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []

    def _record(self, operation: str, details: dict) -> StoreEvent:
        event = StoreEvent(operation=operation, details=details)
        with self._lock:
            self._events.append(event)
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass
        return event

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subscribers.remove(q)

    def get_events(self) -> list[StoreEvent]:
        with self._lock:
            return list(self._events)

    # --- MemoryStore delegation ---

    def add(self, text: str, metadata: dict | None = None, id: str | None = None) -> str:
        mid = self.inner.add(text, metadata=metadata, id=id)
        self._record("add", {"text": text, "id": mid, "metadata": metadata or {}})
        return mid

    def add_batch(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        result_ids = self.inner.add_batch(texts, metadatas=metadatas, ids=ids)
        self._record("add_batch", {"count": len(texts), "ids": result_ids})
        return result_ids

    def search(
        self, query: str, n: int = 10, filters: dict | None = None
    ) -> list[SearchResult]:
        results = self.inner.search(query, n=n, filters=filters)
        self._record(
            "search",
            {
                "query": query,
                "n_results": len(results),
                "results": [
                    {"id": r.entry.id, "text": r.entry.text, "distance": r.distance}
                    for r in results
                ],
            },
        )
        return results

    def delete(self, id: str) -> None:
        self.inner.delete(id)
        self._record("delete", {"id": id})

    def update(
        self, id: str, text: str | None = None, metadata: dict | None = None
    ) -> None:
        self.inner.update(id, text=text, metadata=metadata)
        self._record("update", {"id": id, "text": text, "metadata": metadata})

    def get(self, id: str) -> MemoryEntry | None:
        return self.inner.get(id)

    def get_all(self, filters: dict | None = None) -> list[MemoryEntry]:
        return self.inner.get_all(filters=filters)

    def count(self, filters: dict | None = None) -> int:
        return self.inner.count(filters=filters)
