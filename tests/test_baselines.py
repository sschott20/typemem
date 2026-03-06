"""Tests for full-context and monolithic RAG baselines."""
import pytest

from typemem.baselines import make_full_context, make_monolithic_rag


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
