"""Base implementation of InjectionPlugin with default retrieval + scoring + truncation.

Subclass this to customize specific steps while reusing the common pipeline.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from typemem.memory_item import MemoryItem, MemoryTier
from typemem.plugins.base import InjectionPlugin


@dataclass
class InjectionSpec:
    """Configuration for the default injection pipeline.

    Per-stage filtering is tag-based:
      - included_tags: if non-empty, an item must contain at least one of these
        tags to be eligible (allowlist). Empty = no allowlist.
      - excluded_tags: an item containing any of these tags is dropped (denylist).

    Tags include automatic ones (e.g. ``tier:M1``, ``type:lesson``,
    ``source:scene_graph``) and any plugin-added ones.

    Subclasses can either set these as class attributes or build them in __init__.
    """
    max_tokens: int = 500
    n_results: int = 10
    recency_weight: float = 0.3
    pinned_sources: List[str] = field(default_factory=list)
    live_sources: List[str] = field(default_factory=list)
    included_tags: List[str] = field(default_factory=list)
    excluded_tags: List[str] = field(default_factory=list)


_LESSON_RULES_HEADER = (
    "[LEARNED RULES — MUST FOLLOW] Validated rules from experience. "
    "Follow unless clearly inapplicable. If deviating, state why."
)


def _classify(item: MemoryItem) -> str:
    """Categorize an item for grouped framing.

    Returns one of: "lesson_rule" (M3 LESSON, gets shared header),
    "tip" (non-M3 lesson/tip, gets per-item TIP prefix), "other".
    """
    from typemem.memory_item import MemoryType
    is_lesson = item.memory_type == MemoryType.LESSON
    is_tip_source = item.source == "tip"
    if is_lesson and item.tier == MemoryTier.M3:
        return "lesson_rule"
    if is_lesson or is_tip_source:
        return "tip"
    return "other"


def _format_memory_line(item: MemoryItem) -> str:
    """Default framing for memory items. Subclasses can override format_item().

    Lesson rules are emitted as bare bullet bodies — the shared
    `[LEARNED RULES — MUST FOLLOW]` header is added once at the assembly
    stage in build_context() to avoid repetition.
    """
    kind = _classify(item)
    if kind == "lesson_rule":
        return f"- {item.document}"
    if kind == "tip":
        return f"[TIP — learned from experience, apply if relevant] {item.document}"
    return f"[{item.tier}] {item.document}"


class BaseInjectionPlugin(InjectionPlugin):
    """Default injection pipeline: live sources → pinned → vector search → score → truncate.

    Subclasses override `stage` and `spec` (or the individual pipeline methods
    to customize specific steps).
    """

    # Subclasses must define these
    @property
    def stage(self) -> str:
        raise NotImplementedError

    @property
    def spec(self) -> InjectionSpec:
        """Return the injection specification for this stage."""
        raise NotImplementedError

    # --- Pipeline methods (override to customize) ---

    def get_query(self, query: str, ctx: Optional[Dict[str, Any]]) -> str:
        """Transform the query before vector search. Default: return as-is.
        Override to augment with pose, subtask goal, etc.
        """
        return query

    def get_live_sources(self, ctx: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Collect live summaries from configured live_sources. Always fresh."""
        if self._runner is None:
            return {}
        results: Dict[str, str] = {}
        for name in self.spec.live_sources:
            value = self._runner.get_live_summary(name)
            if value:
                results[name] = value
        return results

    def get_pinned_items(self) -> List[MemoryItem]:
        """Return pinned items — the most recent item from each pinned_source."""
        items: List[MemoryItem] = []
        for src in self.spec.pinned_sources:
            src_items = self._manager.get_by_source(src)
            if not src_items:
                continue
            items.append(max(src_items, key=lambda it: it.timestamp))
        return items

    def search(self, query: str) -> Tuple[List[MemoryItem], List[float]]:
        """Vector search across all tiers, then post-filter by tag policy.

        Oversearches by 3x to leave room for tag filtering, then trims back
        to ``spec.n_results``.
        """
        spec = self.spec
        oversearch = max(spec.n_results * 3, spec.n_results)
        items, distances = self._manager.search_with_distances(
            query=query, tiers=None, n_results=oversearch,
        )
        included = set(spec.included_tags)
        excluded = set(spec.excluded_tags)
        filtered_items: List[MemoryItem] = []
        filtered_dists: List[float] = []
        for item, dist in zip(items, distances):
            if excluded and (item.tags & excluded):
                continue
            if included and not (item.tags & included):
                continue
            filtered_items.append(item)
            filtered_dists.append(dist)
            if len(filtered_items) >= spec.n_results:
                break
        return filtered_items, filtered_dists

    def score(self, items: List[MemoryItem], distances: List[float], now: float) -> List[Tuple[MemoryItem, float]]:
        """Score items by relevance + recency. Returns sorted list (highest first)."""
        w = self.spec.recency_weight
        scored = []
        for item, dist in zip(items, distances):
            relevance = max(0.0, 1.0 - dist)
            age = max(now - item.timestamp, 1.0)
            recency = 1.0 / (1.0 + age / 60.0)
            combined = (1.0 - w) * relevance + w * recency
            scored.append((item, combined))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def format_item(self, item: MemoryItem) -> str:
        """Format a single memory item for injection. Override for custom framing."""
        return _format_memory_line(item)

    def _tokens(self, text: str) -> int:
        """Rough token estimate."""
        return len(text) // 4

    # --- Main entry point ---

    def build_context(self, query: str, ctx: Optional[Dict[str, Any]] = None) -> str:
        spec = self.spec
        effective_max_tokens = spec.max_tokens
        now = time.time()
        header_tokens = self._tokens(_LESSON_RULES_HEADER)

        # 1. Live sources (always fresh)
        live_data = self.get_live_sources(ctx)
        live_lines = list(live_data.values())
        live_results = [
            {"id": None, "text": content, "tier": "live", "score": None, "source": name}
            for name, content in live_data.items()
        ]

        # 2. Pinned items — split into lesson_rules vs others
        pinned_lesson_lines: List[str] = []
        pinned_other_lines: List[str] = []
        pinned_ids = set()
        pinned_tokens = 0
        any_lesson_so_far = False
        for item in self.get_pinned_items():
            pinned_ids.add(item.id)
            line = self.format_item(item)
            line_tokens = self._tokens(line)
            kind = _classify(item)
            extra = header_tokens if (kind == "lesson_rule" and not any_lesson_so_far) else 0
            if (pinned_lesson_lines or pinned_other_lines) \
               and pinned_tokens + line_tokens + extra > effective_max_tokens:
                break
            if kind == "lesson_rule":
                pinned_lesson_lines.append(line)
                any_lesson_so_far = True
            else:
                pinned_other_lines.append(line)
            pinned_tokens += line_tokens + extra

        # 3. Vector search (skip if n_results == 0 — pinned/live only mode)
        search_results: List[Dict[str, Any]] = []
        search_lesson_lines: List[str] = []
        search_other_lines: List[str] = []
        if spec.n_results > 0:
            effective_query = self.get_query(query, ctx)
            items, distances = self.search(effective_query)
            filtered_items: List[MemoryItem] = []
            filtered_dists: List[float] = []
            for item, dist in zip(items, distances):
                if item.id not in pinned_ids:
                    filtered_items.append(item)
                    filtered_dists.append(dist)
            scored = self.score(filtered_items, filtered_dists, now)

            # 4. Truncate to remaining budget
            remaining_tokens = effective_max_tokens - pinned_tokens
            token_count = 0
            for item, score_val in scored:
                line = self.format_item(item)
                line_tokens = self._tokens(line)
                kind = _classify(item)
                extra = header_tokens if (kind == "lesson_rule" and not any_lesson_so_far) else 0
                if token_count + line_tokens + extra > remaining_tokens:
                    break
                if kind == "lesson_rule":
                    search_lesson_lines.append(line)
                    any_lesson_so_far = True
                else:
                    search_other_lines.append(line)
                search_results.append({
                    "id": item.id,
                    "text": line,
                    "tier": item.tier.label,
                    "score": round(score_val, 4),
                    "source": item.source,
                })
                token_count += line_tokens + extra

        # 5. Assemble output. Lessons are grouped under a single shared
        # header (pinned lessons first, then search-derived).
        all_lesson_lines = pinned_lesson_lines + search_lesson_lines
        lesson_block_lines: List[str] = []
        if all_lesson_lines:
            lesson_block_lines.append(_LESSON_RULES_HEADER)
            lesson_block_lines.extend(all_lesson_lines)

        self._last_results = live_results + search_results
        return "\n".join(
            live_lines + lesson_block_lines + pinned_other_lines + search_other_lines
        )
