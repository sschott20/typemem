import logging
import time
from dataclasses import dataclass
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

    def set_recorder(self, recorder):
        self._recorder = recorder

    def set_stage_config(self, stage: str, config: StageConfig):
        self._configs[stage] = config

    def inject(self, stage: str, query: str) -> str:
        config = self._configs.get(stage)
        if config is None:
            logger.warning("No injection config for stage '%s'", stage)
            return ""

        cache_key = (stage, query)
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached is not None:
            cache_time, cached_result = cached
            if now - cache_time < self._cache_ttl:
                return cached_result

        items, distances = self._manager.search_with_distances(
            query=query, tiers=config.tiers, n_results=config.n_results,
        )

        if not items:
            return ""

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
            if token_count + line_tokens > config.max_tokens:
                break
            lines.append(line)
            selected_ids.append(item.id)
            selected_scores.append(round(score, 4))
            token_count += line_tokens

        if self._recorder and selected_ids:
            self._recorder.record_injection(
                stage=stage, memory_ids=selected_ids, scores=selected_scores,
            )

        result = "\n".join(lines)

        self._cache[cache_key] = (now, result)
        if len(self._cache) > self._max_cache_size:
            self._cache = {
                k: v for k, v in self._cache.items()
                if now - v[0] < self._cache_ttl
            }

        return result
