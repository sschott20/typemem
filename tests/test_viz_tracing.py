"""Tests for viz tracing event dataclasses."""
import time

import pytest

from typemem.viz.tracing import ConsolidationEvent, InjectionEvent, StoreEvent


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
