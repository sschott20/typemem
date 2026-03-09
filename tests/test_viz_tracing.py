"""Tests for viz tracing event dataclasses and TracingStore."""
import time
from collections import deque

import pytest

from typemem.viz.tracing import ConsolidationEvent, InjectionEvent, StoreEvent
from typemem.chromadb_store import ChromaDBStore
from typemem.viz.tracing import TracingStore, TracingSystem
from typemem.baselines import make_tiered_memory, make_full_context


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


class TestTracingSystem:
    def test_consolidation_tracked(self, store):
        tracing = TracingStore(store)
        system = make_tiered_memory(tracing, retention_secs=600.0)
        ts = TracingSystem(system, tracing)
        for i in range(5):
            ts.observe({"text": f"Observation {i}"})
        ts.consolidate()
        consol_events = ts.get_consolidations()
        assert len(consol_events) == 1
        assert len(consol_events[0].outputs) >= 1
        assert consol_events[0].duration_ms >= 0

    def test_injection_tracked(self, store):
        tracing = TracingStore(store)
        system = make_full_context(tracing)
        ts = TracingSystem(system, tracing)
        ts.observe({"text": "cup on table"})
        ts.inject("dump", "where is cup", 500)
        inj_events = ts.get_injections()
        assert len(inj_events) == 1
        assert inj_events[0].query == "where is cup"
        assert inj_events[0].token_budget == 500
        assert len(inj_events[0].context) > 0

    def test_delegates_to_inner_system(self, store):
        tracing = TracingStore(store)
        system = make_full_context(tracing)
        ts = TracingSystem(system, tracing)
        ids = ts.observe({"text": "hello"})
        assert len(ids) == 1
        assert tracing.count() == 1

    def test_consolidation_captures_deletions(self, store):
        tracing = TracingStore(store)
        system = make_tiered_memory(tracing, retention_secs=0.0)
        ts = TracingSystem(system, tracing)
        now = time.time()
        for i in range(3):
            tracing.add(f"Old obs {i}", metadata={"_tier": "raw", "_timestamp": now - 1000})
        ts.consolidate()
        consol_events = ts.get_consolidations()
        assert len(consol_events) == 1
        assert len(consol_events[0].deletions) > 0
        assert len(consol_events[0].outputs) >= 1
