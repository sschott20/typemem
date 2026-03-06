"""LLM-powered tiered memory strategy.

Consolidation calls GPT-4o-mini to summarize observations (M1->M2)
and extract lessons (M2->M3). Same _llm_call pattern as generate.py
for easy test mocking.
"""
from __future__ import annotations

import time

from typemem.baselines import _observe_timestamped, _score_results, _budget_join
from typemem.store import MemoryStore
from typemem.system import MemorySystem

# ---------------------------------------------------------------------------
# LLM plumbing (same pattern as generate.py)
# ---------------------------------------------------------------------------

_OPENAI_CLIENT = None


def _get_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI()
    return _OPENAI_CLIENT


def _llm_call(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Make an LLM call. Separated for easy mocking in tests."""
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SUMMARIZE_PROMPT = """\
Summarize these patrol observations into a concise summary (1-2 sentences), \
then list 3-5 keywords.

{observations}

Summary: <your summary>
Keywords: <comma-separated keywords>"""

_LESSON_PROMPT = """\
Based on these patrol summaries, extract one persistent pattern or lesson \
that should be remembered long-term (1-2 sentences), then list 3-5 keywords.

{summaries}

Lesson: <your lesson>
Keywords: <comma-separated keywords>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_response(text: str, label: str = "Summary") -> tuple[str, str]:
    """Parse 'Label: ...\nKeywords: ...' from LLM response."""
    content = ""
    keywords = ""
    for line in text.strip().split("\n"):
        line_stripped = line.strip()
        if line_stripped.startswith(f"{label}:"):
            content = line_stripped[len(label) + 1:].strip()
        elif line_stripped.startswith("Keywords:"):
            keywords = line_stripped[len("Keywords:"):].strip()
    return content or text.strip(), keywords


def _group_similar(
    store: MemoryStore,
    entries: list,
    distance_threshold: float = 0.5,
) -> list[list]:
    """Greedy semantic clustering via store.search().

    Pick an ungrouped entry as seed, find similar entries via search,
    group those within distance_threshold, repeat.
    """
    if not entries:
        return []
    ungrouped = {e.id: e for e in entries}
    groups: list[list] = []
    while ungrouped:
        seed_id = next(iter(ungrouped))
        seed = ungrouped.pop(seed_id)
        group = [seed]
        # Search for similar items in the raw tier
        try:
            results = store.search(seed.text, n=20, filters={"_tier": "raw"})
        except Exception:
            results = []
        for r in results:
            if r.entry.id in ungrouped and r.distance < distance_threshold:
                group.append(ungrouped.pop(r.entry.id))
        groups.append(group)
    return groups


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_tiered_llm(
    store: MemoryStore,
    retention_secs: float = 600.0,
    min_group_size: int = 3,
) -> MemorySystem:
    """Tiered memory with LLM consolidation: raw -> summary -> knowledge.

    Args:
        store: The memory store to use.
        retention_secs: How long (seconds) to keep processed raw entries.
        min_group_size: Minimum items before consolidation triggers.
    """
    system = MemorySystem(store)

    # Processed tracking (closure-captured, no disk I/O)
    processed_raw: set[str] = set()
    processed_summary: set[str] = set()

    # -- Consolidation: M1 -> M2 (summarize) --

    def consolidate_summarize(s: MemoryStore) -> list[str]:
        raw_entries = s.get_all(filters={"_tier": "raw"})
        new_entries = [e for e in raw_entries if e.id not in processed_raw]
        if len(new_entries) < min_group_size:
            return []

        groups = _group_similar(s, new_entries)
        created: list[str] = []
        for group in groups:
            if len(group) < min_group_size:
                continue
            obs_text = "\n".join(f"- {e.text}" for e in group)
            prompt = _SUMMARIZE_PROMPT.format(observations=obs_text)
            response = _llm_call(prompt)
            summary, keywords = _parse_response(response, "Summary")

            sid = s.add(summary, metadata={
                "_tier": "summary",
                "_keywords": keywords,
                "_source": "llm_summarize",
            })
            created.append(sid)
            for e in group:
                processed_raw.add(e.id)
        return created

    # -- Consolidation: M2 -> M3 (extract lessons) --

    def consolidate_extract(s: MemoryStore) -> list[str]:
        summaries = s.get_all(filters={"_tier": "summary"})
        new_summaries = [e for e in summaries if e.id not in processed_summary]
        if len(new_summaries) < min_group_size:
            return []

        created: list[str] = []
        batch_size = 20
        for i in range(0, len(new_summaries), batch_size):
            batch = new_summaries[i:i + batch_size]
            if len(batch) < min_group_size:
                break
            summ_text = "\n".join(f"- {e.text}" for e in batch)
            prompt = _LESSON_PROMPT.format(summaries=summ_text)
            response = _llm_call(prompt)
            lesson, keywords = _parse_response(response, "Lesson")

            kid = s.add(lesson, metadata={
                "_tier": "knowledge",
                "_keywords": keywords,
                "_source": "llm_extract",
            })
            created.append(kid)
            for e in batch:
                processed_summary.add(e.id)
        return created

    # -- Consolidation: prune stale raw entries --

    def consolidate_prune(s: MemoryStore) -> list[str]:
        now = time.time()
        raw_entries = s.get_all(filters={"_tier": "raw"})
        for e in raw_entries:
            if e.id in processed_raw and (now - e.timestamp) > retention_secs:
                s.delete(e.id)
                processed_raw.discard(e.id)
        return []

    # -- Injection: relevance x recency with tier boost --

    def inject_tiered(query: str, s: MemoryStore, token_budget: int) -> str:
        results = s.search(query, n=50)
        scored = _score_results(results, boost_tiers=True)
        return _budget_join((entry.text for _, entry in scored), token_budget)

    # -- Wire up --

    system.add_observation("timestamped", _observe_timestamped, interval=1.0)
    system.add_consolidation("summarize", consolidate_summarize, interval=10.0)
    system.add_consolidation("extract", consolidate_extract, interval=30.0)
    system.add_consolidation("prune", consolidate_prune, interval=60.0)
    system.add_injection("tiered", inject_tiered)
    return system
