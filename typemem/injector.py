import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from typemem.memory_item import MemoryTier
from typemem.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    tiers: List[MemoryTier]
    max_tokens: int = 500
    n_results: int = 10
    recency_weight: float = 0.3
    pinned_sources: List[str] = field(default_factory=list)


DEFAULT_STAGE_CONFIGS: Dict[str, StageConfig] = {
    "S1": StageConfig(tiers=[MemoryTier.M1, MemoryTier.M2], max_tokens=400, n_results=10, recency_weight=0.5),
    "S2": StageConfig(tiers=[MemoryTier.M2, MemoryTier.M3], max_tokens=600, n_results=15, recency_weight=0.3),
    "S3": StageConfig(tiers=[MemoryTier.M2, MemoryTier.M3], max_tokens=600, n_results=15, recency_weight=0.1),
}

_DEFAULT_CACHE_TTL = 5.0


class MemoryInjector:
    """Builds memory context for each stage's prompt. Latency-critical — no LLM calls."""

    def __init__(self, manager: MemoryManager, cache_ttl: float = _DEFAULT_CACHE_TTL):
        self._manager = manager
        self._configs: Dict[str, StageConfig] = dict(DEFAULT_STAGE_CONFIGS)
        self._recorder = None
        self._cache_ttl = cache_ttl
        self._cache: Dict[Tuple[str, str], Tuple[float, str]] = {}
        self._max_cache_size = 100
        self._last_results: List[Dict] = []

    @property
    def last_results(self) -> List[Dict]:
        """Detailed results from the most recent inject() call."""
        return list(self._last_results)

    def set_recorder(self, recorder):
        self._recorder = recorder

    def set_stage_config(self, stage: str, config: StageConfig):
        self._configs[stage] = config

    def inject(self, stage: str, query: str, max_tokens: Optional[int] = None) -> str:
        config = self._configs.get(stage)
        if config is None:
            logger.warning("No injection config for stage '%s'", stage)
            return ""

        effective_max_tokens = max_tokens if max_tokens is not None else config.max_tokens

        cache_key = (stage, query)
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached is not None:
            cache_time, cached_result = cached
            if now - cache_time < self._cache_ttl:
                return cached_result

        # Pinned items — always included, prepended before search results
        pinned_lines = []
        pinned_tokens = 0
        for src in config.pinned_sources:
            src_items = self._manager.get_by_source(src)
            if not src_items:
                continue
            most_recent = max(src_items, key=lambda it: it.timestamp)
            line = f"[{most_recent.tier}] {most_recent.document}"
            line_tokens = len(line) // 4
            # Always include first pinned item; stop adding more if budget exceeded
            if pinned_lines and pinned_tokens + line_tokens > effective_max_tokens:
                break
            pinned_lines.append(line)
            pinned_tokens += line_tokens

        items, distances = self._manager.search_with_distances(
            query=query, tiers=config.tiers, n_results=config.n_results,
        )

        if not items and not pinned_lines:
            return ""

        remaining_tokens = effective_max_tokens - pinned_tokens

        w = config.recency_weight
        scored = []
        for i, item in enumerate(items):
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
            line = f"[{item.tier}] {item.document}"
            line_tokens = len(line) // 4
            if token_count + line_tokens > remaining_tokens:
                break
            lines.append(line)
            selected_ids.append(item.id)
            selected_scores.append(round(score, 4))
            token_count += line_tokens

        self._last_results = [
            {"id": selected_ids[i], "text": lines[i], "tier": scored[i][0].tier.label,
             "score": selected_scores[i]}
            for i in range(len(selected_ids))
        ]

        if self._recorder and selected_ids:
            self._recorder.record_injection(
                stage=stage, memory_ids=selected_ids, scores=selected_scores,
            )

        result = "\n".join(pinned_lines + lines)

        self._cache[cache_key] = (now, result)
        if len(self._cache) > self._max_cache_size:
            self._cache = {
                k: v for k, v in self._cache.items()
                if now - v[0] < self._cache_ttl
            }

        return result
