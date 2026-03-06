"""Baseline memory strategies for comparison."""
from __future__ import annotations

from typing import Iterable

from typemem.store import MemoryStore
from typemem.system import MemorySystem


def _observe_raw(raw_data: dict, store: MemoryStore) -> list[str]:
    text = raw_data.get("text")
    if not text:
        return []
    return [store.add(text)]


def _consolidate_noop(store: MemoryStore) -> list[str]:
    return []


def _budget_join(texts: Iterable[str], token_budget: int) -> str:
    """Join texts until the approximate token budget is exhausted."""
    lines: list[str] = []
    budget_left = token_budget
    for text in texts:
        cost = len(text) // 4
        if cost > budget_left:
            break
        lines.append(text)
        budget_left -= cost
    return "\n".join(lines)


def make_full_context(store: MemoryStore) -> MemorySystem:
    """Full-context baseline: stores raw text, dumps everything on injection."""
    system = MemorySystem(store)

    def inject_dump(query: str, s: MemoryStore, token_budget: int) -> str:
        return _budget_join((e.text for e in s.get_all()), token_budget)

    system.add_observation("raw", _observe_raw, interval=1.0)
    system.add_consolidation("noop", _consolidate_noop, interval=1.0)
    system.add_injection("dump", inject_dump)
    return system


def make_monolithic_rag(store: MemoryStore) -> MemorySystem:
    """Monolithic RAG baseline: stores raw text, retrieves top-k on injection."""
    system = MemorySystem(store)

    def inject_topk(query: str, s: MemoryStore, token_budget: int) -> str:
        return _budget_join((r.entry.text for r in s.search(query, n=50)), token_budget)

    system.add_observation("raw", _observe_raw, interval=1.0)
    system.add_consolidation("noop", _consolidate_noop, interval=1.0)
    system.add_injection("topk", inject_topk)
    return system
