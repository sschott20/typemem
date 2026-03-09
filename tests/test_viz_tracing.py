"""Tests for viz tracing event dataclasses and TracingStore."""
import time
from collections import deque

import pytest

from typemem.viz.tracing import ConsolidationEvent, InjectionEvent, StoreEvent
from typemem.chromadb_store import ChromaDBStore
from typemem.viz.tracing import TracingStore


def test_store_event_creation():
    e = StoreEvent(operation="add", details={"text": "hello", "id": "abc"})
    assert e.operation == "add"
    assert e.details["text"] == "hello"
    assert e.timestamp <= time.time()
    assert e.timestamp > 0


def test_consolidation_event_creation():
    e = ConsolidationEvent(
        name="summarize",
        inputs=[{"id": "a", "text": "raw 1"}],
        outputs=[{"id": "b", "text": "[Summary] raw 1"}],
        deletions=["old1"],
        duration_ms=12.5,
    )
    assert e.name == "summarize"
    assert len(e.inputs) == 1
    assert len(e.deletions) == 1


def test_injection_event_creation():
    e = InjectionEvent(
        name="tiered",
        query="where is the cup?",
        token_budget=500,
        search_results=[{"id": "a", "distance": 0.2, "text": "cup on table"}],
        context="cup on table",
        duration_ms=150.0,
    )
    assert e.query == "where is the cup?"
    assert e.duration_ms == 150.0


@pytest.fixture
def store(tmp_path):
    return ChromaDBStore(persist_dir=str(tmp_path / "chroma"))


@pytest.fixture
def tracing(store):
    return TracingStore(store)


class TestTracingStore:
    def test_add_records_event(self, tracing):
        mid = tracing.add("hello world")
        assert tracing.inner.get(mid) is not None
        events = tracing.get_events()
        assert len(events) == 1
        assert events[0].operation == "add"
        assert events[0].details["text"] == "hello world"
        assert events[0].details["id"] == mid

    def test_add_batch_records_event(self, tracing):
        ids = tracing.add_batch(["a", "b", "c"])
        assert len(ids) == 3
        events = tracing.get_events()
        assert len(events) == 1
        assert events[0].operation == "add_batch"
        assert events[0].details["count"] == 3

    def test_delete_records_event(self, tracing):
        mid = tracing.add("to delete")
        tracing.delete(mid)
        events = tracing.get_events()
        assert events[-1].operation == "delete"
        assert events[-1].details["id"] == mid

    def test_search_records_event(self, tracing):
        tracing.add("the cup is on the table")
        results = tracing.search("cup", n=5)
        events = tracing.get_events()
        search_events = [e for e in events if e.operation == "search"]
        assert len(search_events) == 1
        assert search_events[0].details["query"] == "cup"
        assert search_events[0].details["n_results"] == len(results)

    def test_delegates_all_methods(self, tracing, store):
        """TracingStore passes through to inner store correctly."""
        mid = tracing.add("test entry", metadata={"key": "val"})
        entry = tracing.get(mid)
        assert entry is not None
        assert entry.text == "test entry"
        assert tracing.count() == 1
        all_entries = tracing.get_all()
        assert len(all_entries) == 1
        tracing.update(mid, text="updated")
        assert tracing.get(mid).text == "updated"

    def test_event_deque_bounded(self, tracing):
        """Events deque respects max size."""
        tracing._max_events = 5
        tracing._events = deque(maxlen=5)
        for i in range(10):
            tracing.add(f"entry {i}")
        assert len(tracing.get_events()) == 5

    def test_sse_subscribers_notified(self, tracing):
        """SSE subscribers receive new events."""
        q = tracing.subscribe()
        tracing.add("hello")
        assert not q.empty()
        event = q.get_nowait()
        assert event.operation == "add"
        tracing.unsubscribe(q)
