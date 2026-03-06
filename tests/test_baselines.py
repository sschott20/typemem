"""Tests for full-context and monolithic RAG baselines."""
import time

import pytest

from typemem.baselines import (
    make_full_context,
    make_monolithic_rag,
    make_tiered_memory,
    make_rag_with_recency,
    make_tiered_no_consolidation,
)


@pytest.fixture
def full_context_system(store):
    return make_full_context(store)


@pytest.fixture
def rag_system(store):
    return make_monolithic_rag(store)


class TestFullContext:
    def test_observe_stores_raw(self, full_context_system, store):
        raw = {"text": "Robot sees a cup on the table at position [1,2,3]"}
        ids = full_context_system.observe(raw)
        assert len(ids) == 1
        entry = store.get(ids[0])
        assert "cup" in entry.text.lower()

    def test_consolidation_is_noop(self, full_context_system, store):
        store.add("some memory")
        ids = full_context_system.consolidate()
        assert ids == []

    def test_inject_dumps_everything(self, full_context_system, store):
        store.add("Memory A")
        store.add("Memory B")
        store.add("Memory C")
        context = full_context_system.inject("dump", "any query", token_budget=5000)
        assert "Memory A" in context
        assert "Memory B" in context
        assert "Memory C" in context

    def test_inject_respects_token_budget(self, full_context_system, store):
        for i in range(20):
            store.add(f"This is memory number {i:03d} with some text")
        context = full_context_system.inject("dump", "query", token_budget=50)
        lines = [l for l in context.strip().split("\n") if l.strip()]
        assert len(lines) < 20


class TestMonolithicRAG:
    def test_observe_stores(self, rag_system, store):
        raw = {"text": "Cup on table"}
        ids = rag_system.observe(raw)
        assert len(ids) == 1

    def test_consolidation_is_noop(self, rag_system, store):
        store.add("data")
        ids = rag_system.consolidate()
        assert ids == []

    def test_inject_returns_relevant(self, rag_system, store):
        store.add("The red cup is on the kitchen table")
        store.add("The robot battery is at 80%")
        store.add("A blue plate is next to the cup")
        context = rag_system.inject("topk", "where is the cup?", token_budget=500)
        assert "cup" in context.lower()

    def test_inject_respects_token_budget(self, rag_system, store):
        for i in range(50):
            store.add(f"Memory item number {i:03d} with padding text here")
        context = rag_system.inject("topk", "query", token_budget=30)
        lines = [l for l in context.strip().split("\n") if l.strip()]
        assert len(lines) < 50


# ---------------------------------------------------------------------------
# Tiered memory
# ---------------------------------------------------------------------------

@pytest.fixture
def tiered_system(store):
    return make_tiered_memory(store, retention_secs=600.0)


class TestTieredMemory:
    def test_observe_adds_tier_metadata(self, tiered_system, store):
        ids = tiered_system.observe({"text": "Robot picked up a cup"})
        assert len(ids) == 1
        entry = store.get(ids[0])
        assert entry.metadata.get("_tier") == "raw"

    def test_consolidation_creates_summaries(self, tiered_system, store):
        for i in range(5):
            tiered_system.observe({"text": f"Observation {i}"})
        created = tiered_system.consolidate()
        assert len(created) >= 1
        summary_entry = store.get(created[0])
        assert summary_entry.metadata.get("_tier") == "summary"
        assert summary_entry.text.startswith("[Summary]")

    def test_injection_returns_context(self, tiered_system, store):
        for i in range(3):
            tiered_system.observe({"text": f"The robot sees item {i}"})
        context = tiered_system.inject("tiered", "what does the robot see", token_budget=500)
        assert len(context) > 0

    def test_retention_deletes_old_raw(self, store):
        # Use a very short retention so we can test deletion without real sleep
        system = make_tiered_memory(store, retention_secs=0.0)
        now = time.time()
        # Add 3 entries with old timestamps (via store directly, to control _timestamp)
        for i in range(3):
            store.add(
                f"Old observation {i}",
                metadata={"_tier": "raw", "_timestamp": now - 1000},
            )
        # First consolidation: processes the 3 raw entries, creates summary,
        # and deletes old ones (retention_secs=0 means all are "old").
        system.consolidate()
        remaining_raw = store.get_all(filters={"_tier": "raw"})
        assert len(remaining_raw) == 0
        # Summary should still exist
        summaries = store.get_all(filters={"_tier": "summary"})
        assert len(summaries) == 1
