"""Baseline memory strategies for comparison."""
from __future__ import annotations

from typemem.store import MemoryStore
from typemem.system import MemorySystem


def make_full_context(store: MemoryStore) -> MemorySystem:
    """Full-context baseline: stores raw text, dumps everything on injection."""
    system = MemorySystem(store)

    def observe_raw(raw_data: dict, s: MemoryStore) -> list[str]:
        text = raw_data.get("text")
        if not text:
            return []
        mid = s.add(text)
        return [mid]

    def consolidate_noop(s: MemoryStore) -> list[str]:
        return []

    def inject_dump(query: str, s: MemoryStore, token_budget: int) -> str:
        entries = s.get_all()
        lines: list[str] = []
        budget_left = token_budget
        for entry in entries:
            cost = len(entry.text) // 4
            if cost > budget_left:
                break
            lines.append(entry.text)
            budget_left -= cost
        return "\n".join(lines)

    system.add_observation("raw", observe_raw, interval=1.0)
    system.add_consolidation("noop", consolidate_noop, interval=1.0)
    system.add_injection("dump", inject_dump)
    return system


def make_monolithic_rag(store: MemoryStore) -> MemorySystem:
    """Monolithic RAG baseline: stores raw text, retrieves top-k on injection."""
    system = MemorySystem(store)

    def observe_raw(raw_data: dict, s: MemoryStore) -> list[str]:
        text = raw_data.get("text")
        if not text:
            return []
        mid = s.add(text)
        return [mid]

    def consolidate_noop(s: MemoryStore) -> list[str]:
        return []

    def inject_topk(query: str, s: MemoryStore, token_budget: int) -> str:
        results = s.search(query, n=50)
        lines: list[str] = []
        budget_left = token_budget
        for result in results:
            cost = len(result.entry.text) // 4
            if cost > budget_left:
                break
            lines.append(result.entry.text)
            budget_left -= cost
        return "\n".join(lines)

    system.add_observation("raw", observe_raw, interval=1.0)
    system.add_consolidation("noop", consolidate_noop, interval=1.0)
    system.add_injection("topk", inject_topk)
    return system
