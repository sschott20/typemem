"""Tracing event dataclasses and TracingStore for memory visualization."""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from typemem.store import MemoryStore
from typemem.system import MemorySystem
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

    def broadcast(self, event: Any) -> None:
        """Push an arbitrary event to all SSE subscribers."""
        with self._lock:
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

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


class TracingSystem:
    """Wrapper around MemorySystem that captures consolidation and injection details."""

    def __init__(self, system: MemorySystem, tracing_store: TracingStore):
        self.system = system
        self.tracing_store = tracing_store
        self._consolidations: deque[ConsolidationEvent] = deque(maxlen=1000)
        self._injections: deque[InjectionEvent] = deque(maxlen=1000)
        self._lock = threading.Lock()

    def observe(self, raw_data: dict) -> list[str]:
        return self.system.observe(raw_data)

    def consolidate(self) -> list[str]:
        # Snapshot entries before consolidation to identify inputs
        pre_entries = self.tracing_store.inner.get_all()
        pre_snapshot = {
            e.id: {"id": e.id, "text": e.text, "metadata": e.metadata, "timestamp": e.timestamp}
            for e in pre_entries
        }

        events_before = len(self.tracing_store.get_events())
        t0 = time.perf_counter()
        ids = self.system.consolidate()
        duration_ms = (time.perf_counter() - t0) * 1000.0

        all_events = self.tracing_store.get_events()
        new_events = all_events[events_before:]

        outputs = []
        deletions = []
        searched_ids: set[str] = set()
        for ev in new_events:
            if ev.operation == "add":
                outputs.append({"id": ev.details["id"], "text": ev.details["text"],
                                "metadata": ev.details.get("metadata", {})})
            elif ev.operation == "add_batch":
                for mid in ev.details.get("ids", []):
                    outputs.append({"id": mid})
            elif ev.operation == "delete":
                deletions.append(ev.details["id"])
            elif ev.operation == "search":
                for r in ev.details.get("results", []):
                    if r.get("id"):
                        searched_ids.add(r["id"])

        # Inputs: entries that were accessed during consolidation
        # 1. Entries returned by search calls (semantic clustering)
        # 2. Entries that were deleted (pruned)
        # 3. If no searches happened but outputs were created, diff to find consumed entries
        inputs = []
        seen_ids: set[str] = set()
        if outputs or deletions:
            # Searched entries that existed before = direct inputs to summarization
            for eid in searched_ids:
                if eid in pre_snapshot and eid not in seen_ids:
                    # Exclude entries that were created during this consolidation
                    output_ids = {o.get("id") for o in outputs}
                    if eid not in output_ids:
                        inputs.append(pre_snapshot[eid])
                        seen_ids.add(eid)
            # Deleted entries that weren't already captured
            for did in deletions:
                if did in pre_snapshot and did not in seen_ids:
                    inputs.append(pre_snapshot[did])
                    seen_ids.add(did)
            # Fallback: if no searches but outputs exist (simple concatenation strategy),
            # find entries whose text appears in output text
            if not searched_ids and outputs:
                output_texts = " ".join(o.get("text", "") for o in outputs)
                for entry in pre_snapshot.values():
                    if entry["id"] not in seen_ids and entry["text"] in output_texts:
                        inputs.append(entry)
                        seen_ids.add(entry["id"])

        event = ConsolidationEvent(
            name=self._guess_consolidation_name(),
            inputs=inputs,
            outputs=outputs,
            deletions=deletions,
            duration_ms=duration_ms,
        )
        with self._lock:
            self._consolidations.append(event)
        self.tracing_store.broadcast(event)
        return ids

    def inject(self, name: str, query: str, token_budget: int) -> str:
        events_before = len(self.tracing_store.get_events())
        t0 = time.perf_counter()
        context = self.system.inject(name, query, token_budget)
        duration_ms = (time.perf_counter() - t0) * 1000.0

        all_events = self.tracing_store.get_events()
        new_events = all_events[events_before:]
        search_results = []
        for ev in new_events:
            if ev.operation == "search":
                search_results = ev.details.get("results", [])
                break

        event = InjectionEvent(
            name=name,
            query=query,
            token_budget=token_budget,
            search_results=search_results,
            context=context,
            duration_ms=duration_ms,
        )
        with self._lock:
            self._injections.append(event)
        self.tracing_store.broadcast(event)
        return context

    def get_consolidations(self) -> list[ConsolidationEvent]:
        with self._lock:
            return list(self._consolidations)

    def get_injections(self) -> list[InjectionEvent]:
        with self._lock:
            return list(self._injections)

    def _guess_consolidation_name(self) -> str:
        names = list(self.system._consolidations.keys())
        return names[0] if names else "unknown"

    # Pass-through for MemorySystem API
    def add_observation(self, *a, **kw):
        self.system.add_observation(*a, **kw)

    def add_consolidation(self, *a, **kw):
        self.system.add_consolidation(*a, **kw)

    def add_injection(self, *a, **kw):
        self.system.add_injection(*a, **kw)

    def start(self, *a, **kw):
        self.system.start(*a, **kw)

    def stop(self):
        self.system.stop()
