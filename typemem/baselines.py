"""Baseline memory strategies for comparison."""
from __future__ import annotations

import time
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


# ---------------------------------------------------------------------------
# Recency-weighted scoring (shared by tiered and recency-only injection)
# ---------------------------------------------------------------------------

def _score_results(
    results: list,
    boost_tiers: bool,
    max_age: float = 3600.0,
) -> list[tuple[float, "MemoryEntry"]]:
    """Score search results by relevance + recency, optionally with tier boost."""
    from typemem.types import SearchResult  # noqa: local import to avoid cycles

    now = time.time()
    scored: list[tuple[float, object]] = []
    for r in results:
        recency = max(0.0, 1.0 - (now - r.entry.timestamp) / max_age)
        score = (1.0 - r.distance) * 0.7 + recency * 0.3
        if boost_tiers:
            tier = r.entry.metadata.get("_tier", "raw")
            if tier == "summary":
                score *= 1.5
            elif tier == "knowledge":
                score *= 2.0
        scored.append((score, r.entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Tiered memory strategy
# ---------------------------------------------------------------------------

def _observe_timestamped(raw_data: dict, store: MemoryStore) -> list[str]:
    """Observation that tags entries with _tier='raw'."""
    text = raw_data.get("text")
    if not text:
        return []
    return [store.add(text, metadata={"_tier": "raw"})]


def make_tiered_memory(store: MemoryStore, retention_secs: float = 600.0) -> MemorySystem:
    """Tiered memory: raw -> summary -> knowledge with recency + tier boosting."""
    system = MemorySystem(store)

    processed_ids: set[str] = set()

    def consolidate_summarize(s: MemoryStore) -> list[str]:
        raw_entries = s.get_all(filters={"_tier": "raw"})
        new_entries = [e for e in raw_entries if e.id not in processed_ids]

        created_ids: list[str] = []
        if len(new_entries) >= 3:
            for e in new_entries:
                processed_ids.add(e.id)
            summary_text = "[Summary] " + "; ".join(e.text for e in new_entries)
            sid = s.add(summary_text, metadata={"_tier": "summary"})
            created_ids.append(sid)

        # Retention: delete old processed raw entries
        now = time.time()
        for e in raw_entries:
            if e.id in processed_ids and (now - e.timestamp) > retention_secs:
                s.delete(e.id)
                processed_ids.discard(e.id)

        return created_ids

    def inject_tiered(query: str, s: MemoryStore, token_budget: int) -> str:
        results = s.search(query, n=50)
        scored = _score_results(results, boost_tiers=True)
        return _budget_join((entry.text for _, entry in scored), token_budget)

    system.add_observation("timestamped", _observe_timestamped, interval=1.0)
    system.add_consolidation("summarize", consolidate_summarize, interval=10.0)
    system.add_injection("tiered", inject_tiered)
    return system


# ---------------------------------------------------------------------------
# Ablation variants
# ---------------------------------------------------------------------------

def make_rag_with_recency(store: MemoryStore) -> MemorySystem:
    """Monolithic RAG with recency-weighted scoring (no tier boosting)."""
    system = MemorySystem(store)

    def inject_recency(query: str, s: MemoryStore, token_budget: int) -> str:
        results = s.search(query, n=50)
        scored = _score_results(results, boost_tiers=False)
        return _budget_join((entry.text for _, entry in scored), token_budget)

    system.add_observation("raw", _observe_raw, interval=1.0)
    system.add_consolidation("noop", _consolidate_noop, interval=1.0)
    system.add_injection("recency", inject_recency)
    return system


def make_tiered_no_consolidation(store: MemoryStore) -> MemorySystem:
    """Tiered observation + injection but consolidation is noop."""
    system = MemorySystem(store)

    def inject_tiered(query: str, s: MemoryStore, token_budget: int) -> str:
        results = s.search(query, n=50)
        scored = _score_results(results, boost_tiers=True)
        return _budget_join((entry.text for _, entry in scored), token_budget)

    system.add_observation("timestamped", _observe_timestamped, interval=1.0)
    system.add_consolidation("noop", _consolidate_noop, interval=1.0)
    system.add_injection("tiered", inject_tiered)
    return system
