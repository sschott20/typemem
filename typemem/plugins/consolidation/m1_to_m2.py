"""Consolidation plugin: M1 -> M2 (observations -> summaries via LLM)."""

import logging
from typing import List, Optional

from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex
from typemem.utils import parse_summary_keywords
from typemem.plugins.base import ConsolidationPlugin
from typemem.plugins.consolidation._utils import most_common_waypoint, group_by_similarity

logger = logging.getLogger(__name__)


class M1ToM2Strategy(ConsolidationPlugin):
    """Summarize groups of similar M1 observations into M2 summaries using LLM."""

    def __init__(self, min_group_size: int = 3, distance_threshold: float = 0.5):
        self._min_group_size = min_group_size
        self._distance_threshold = distance_threshold

    @property
    def name(self) -> str:
        return "M1ToM2"

    @property
    def source_tier(self) -> MemoryTier:
        return MemoryTier.M1

    @property
    def target_tier(self) -> MemoryTier:
        return MemoryTier.M2

    @property
    def interval_seconds(self) -> float:
        return 30.0

    def run(self, manager: MemoryManager, llm=None, processed_index: Optional[ProcessedIndex] = None) -> List[str]:
        if llm is None:
            logger.debug("M1ToM2: skipping, no LLM available")
            return []

        unprocessed = self.get_unprocessed(manager, processed_index, tiers=[self.source_tier] if self.source_tier else None) if processed_index else manager.get_by_tier(self.source_tier)

        if len(unprocessed) < self._min_group_size:
            logger.debug("M1ToM2: skipping, only %d unprocessed (need %d)", len(unprocessed), self._min_group_size)
            return []

        groups = group_by_similarity(manager, unprocessed, MemoryTier.M1,
                                     distance_threshold=self._distance_threshold)

        new_ids = []
        for group in groups:
            if len(group) < self._min_group_size:
                continue

            observations_text = "\n".join([f"- {item.document}" for item in group])
            prompt = (
                "Summarize these robot observations into a concise summary, "
                "then list 3-5 keywords that capture the key concepts.\n\n"
                f"{observations_text}\n\n"
                "Summary: <one or two sentence summary>\n"
                "Keywords: <comma-separated keywords>"
            )

            try:
                response = llm(prompt)
                summary, keywords = parse_summary_keywords(response)
            except Exception as e:
                logger.error("LLM summarization failed for M1->M2: %s", e)
                continue

            m2_item = MemoryItem(
                document=summary,
                tier=MemoryTier.M2,
                memory_type=MemoryType.SUMMARY,
                robot_id=group[0].robot_id,
                waypoint=most_common_waypoint(group),
                keywords=keywords,
                source="M1ToM2",
            )
            new_id = manager.add(m2_item)
            new_ids.append(new_id)

            if processed_index:
                self.mark_done(processed_index, [item.id for item in group])

        if new_ids:
            logger.info("M1->M2: created %d summaries from %d observations", len(new_ids), len(unprocessed))
        return new_ids
