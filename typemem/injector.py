import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from typing import TYPE_CHECKING

from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager

if TYPE_CHECKING:
    from typemem.plugins.runner import ObservationRunner

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    tiers: List[MemoryTier]
    max_tokens: int = 500
    n_results: int = 10
    recency_weight: float = 0.3
    pinned_sources: List[str] = field(default_factory=list)
    live_sources: List[str] = field(default_factory=list)


DEFAULT_STAGE_CONFIGS: Dict[str, StageConfig] = {
    "S1": StageConfig(tiers=[MemoryTier.M1, MemoryTier.M2], max_tokens=400, n_results=10, recency_weight=0.5, live_sources=[]),
    "S2": StageConfig(tiers=[MemoryTier.M2, MemoryTier.M3], max_tokens=600, n_results=15, recency_weight=0.3, live_sources=[]),
    "S3": StageConfig(tiers=[MemoryTier.M2, MemoryTier.M3], max_tokens=600, n_results=15, recency_weight=0.1, live_sources=[]),
}

_DEFAULT_CACHE_TTL = 5.0


def _format_memory_line(item: MemoryItem) -> str:
    """Consistent framing for memory items in injected context."""
    is_lesson = item.memory_type == MemoryType.LESSON
    is_tip = item.source == "tip"

    if is_lesson and item.tier == MemoryTier.M3:
        return (
            "[LEARNED RULES — MUST FOLLOW] "
            "Validated rules from experience. Follow unless clearly inapplicable. "
            "If deviating, state why.\n"
            + item.document
        )
    if is_lesson or is_tip:
        return f"[TIP — learned from experience, apply if relevant] {item.document}"
    return f"[{item.tier}] {item.document}"


class MemoryInjector:
    """Builds memory context for each stage's prompt. Latency-critical — no LLM calls."""

    def __init__(self, manager: MemoryManager, cache_ttl: float = _DEFAULT_CACHE_TTL):
        self._manager = manager
        self._configs: Dict[str, StageConfig] = dict(DEFAULT_STAGE_CONFIGS)
        self._recorder = None
        self._on_injection = None  # callback: (stage, query, context, results) -> None
        self._cache_ttl = cache_ttl
        self._cache: Dict[Tuple[str, str], Tuple[float, str]] = {}
        self._max_cache_size = 100
        self._last_results: List[Dict] = []
        self._runner: Optional["ObservationRunner"] = None

    @property
    def last_results(self) -> List[Dict]:
        """Detailed results from the most recent inject() call."""
        return list(self._last_results)

    def set_recorder(self, recorder):
        self._recorder = recorder

    def set_on_injection(self, callback):
        """Set a callback invoked after each inject(): callback(stage, query, context, results)."""
        self._on_injection = callback

    def set_stage_config(self, stage: str, config: StageConfig):
        self._configs[stage] = config

    def set_runner(self, runner: "ObservationRunner") -> None:
        """Set the ObservationRunner used for plugin-based live sources."""
        self._runner = runner

    def get_live_sources(self) -> Dict[str, str]:
        """Get all live summaries from the observation runner. Used by viz server."""
        if self._runner is None:
            return {}
        return self._runner.get_all_live_summaries()

    def inject(self, stage: str, query: str, max_tokens: Optional[int] = None) -> str:
        config = self._configs.get(stage)
        if config is None:
            logger.warning("No injection config for stage '%s'", stage)
            return ""

        effective_max_tokens = max_tokens if max_tokens is not None else config.max_tokens

        # Collect live sources — per-stage, always fresh, bypass cache
        live_names = config.live_sources
        has_live = bool(live_names) and self._runner is not None
        live_data = {}
        if has_live:
            for name in live_names:
                value = self._runner.get_live_summary(name)
                if value:
                    live_data[name] = value

        live_lines = list(live_data.values())
        live_results = [
            {"id": None, "text": content, "tier": "live", "score": None, "source": name}
            for name, content in live_data.items()
        ]

        cache_key = (stage, query)
        now = time.time()
        if not has_live:
            cached = self._cache.get(cache_key)
            if cached is not None:
                cache_time, cached_result = cached
                if now - cache_time < self._cache_ttl:
                    return cached_result

        # Pinned items — always included, prepended before search results
        pinned_lines = []
        pinned_tokens = 0
        pinned_ids = set()
        for src in config.pinned_sources:
            src_items = self._manager.get_by_source(src)
            if not src_items:
                continue
            most_recent = max(src_items, key=lambda it: it.timestamp)
            pinned_ids.add(most_recent.id)
            line = _format_memory_line(most_recent)
            line_tokens = len(line) // 4
            # Always include first pinned item; stop adding more if budget exceeded
            if pinned_lines and pinned_tokens + line_tokens > effective_max_tokens:
                break
            pinned_lines.append(line)
            pinned_tokens += line_tokens

        items, distances = self._manager.search_with_distances(
            query=query, tiers=config.tiers, n_results=config.n_results,
        )

        if not items and not pinned_lines and not live_lines:
            return ""

        remaining_tokens = effective_max_tokens - pinned_tokens

        w = config.recency_weight
        scored = []
        for i, item in enumerate(items):
            if item.id in pinned_ids:
                continue  # already included as pinned
            relevance = max(0.0, 1.0 - distances[i])
            age = max(now - item.timestamp, 1.0)
            recency = 1.0 / (1.0 + age / 60.0)
            combined = (1.0 - w) * relevance + w * recency
            scored.append((item, combined))

        scored.sort(key=lambda x: x[1], reverse=True)

        lines = []
        selected_ids = []
        selected_scores = []
        token_count = 0
        for item, score in scored:
            line = _format_memory_line(item)
            line_tokens = len(line) // 4
            if token_count + line_tokens > remaining_tokens:
                break
            lines.append(line)
            selected_ids.append(item.id)
            selected_scores.append(round(score, 4))
            token_count += line_tokens

        search_results = [
            {"id": selected_ids[i], "text": lines[i], "tier": scored[i][0].tier.label,
             "score": selected_scores[i]}
            for i in range(len(selected_ids))
        ]
        self._last_results = live_results + search_results

        if self._recorder and selected_ids:
            self._recorder.record_injection(
                stage=stage, memory_ids=selected_ids, scores=selected_scores,
            )

        result = "\n".join(live_lines + pinned_lines + lines)

        if self._on_injection and result:
            try:
                self._on_injection(stage, query, result, self._last_results)
            except Exception:
                pass

        if not has_live:
            self._cache[cache_key] = (now, result)
            if len(self._cache) > self._max_cache_size:
                self._cache = {
                    k: v for k, v in self._cache.items()
                    if now - v[0] < self._cache_ttl
                }

        return result
