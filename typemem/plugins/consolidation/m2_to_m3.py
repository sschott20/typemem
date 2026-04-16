"""Consolidation plugin: M2 -> M3 (summaries -> long-term lessons via LLM)."""

import logging
from typing import List, Optional

from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex
from typemem.utils import parse_summary_keywords
from typemem.plugins.base import ConsolidationPlugin
from typemem.plugins.consolidation._utils import most_common_waypoint

logger = logging.getLogger(__name__)

_MAX_BATCH_SIZE = 20


class M2ToM3Strategy(ConsolidationPlugin):
    """Extract long-term knowledge from M2 summaries using LLM."""

    def __init__(self, min_summaries: int = 3, max_batch_size: int = _MAX_BATCH_SIZE):
        self._min_summaries = min_summaries
        self._max_batch_size = max_batch_size

    @property
    def name(self) -> str:
        return "M2ToM3"

    @property
    def source_tier(self) -> MemoryTier:
        return MemoryTier.M2

    @property
    def target_tier(self) -> MemoryTier:
        return MemoryTier.M3

    @property
    def interval_seconds(self) -> float:
        return 300.0

    def run(self, manager: MemoryManager, llm=None, processed_index: Optional[ProcessedIndex] = None) -> List[str]:
        if llm is None:
            logger.debug("M2ToM3: skipping, no LLM available")
            return []

        unprocessed = self.get_unprocessed(manager, processed_index, tiers=[self.source_tier] if self.source_tier else None) if processed_index else manager.get_by_tier(self.source_tier)

        if len(unprocessed) < self._min_summaries:
            logger.debug("M2ToM3: skipping, only %d unprocessed (need %d)", len(unprocessed), self._min_summaries)
            return []

        # Process in bounded batches
        new_ids = []
        for i in range(0, len(unprocessed), self._max_batch_size):
            batch = unprocessed[i:i + self._max_batch_size]
            result_id = self._process_batch(manager, llm, batch, processed_index)
            if result_id:
                new_ids.append(result_id)

        return new_ids

    def _process_batch(
        self, manager: MemoryManager, llm, batch: List[MemoryItem],
        processed_index: Optional[ProcessedIndex] = None,
    ) -> Optional[str]:
        """Process a single batch of M2 items into one M3 lesson."""
        summaries_text = "\n".join([f"- {item.document}" for item in batch])
        prompt = (
            "Based on these observation summaries, extract one persistent lesson "
            "or pattern that the robot should remember long-term. "
            "Then list 3-5 keywords that capture the key concepts.\n\n"
            f"{summaries_text}\n\n"
            "Lesson: <one or two sentence lesson>\n"
            "Keywords: <comma-separated keywords>"
        )

        try:
            response = llm(prompt)
            lesson, keywords = parse_summary_keywords(response)
        except Exception as e:
            logger.error("LLM lesson extraction failed for M2->M3: %s", e)
            return None

        common_waypoint = most_common_waypoint(batch)

        m3_item = MemoryItem(
            document=lesson,
            tier=MemoryTier.M3,
            memory_type=MemoryType.LESSON,
            robot_id=batch[0].robot_id,
            waypoint=common_waypoint,
            keywords=keywords,
            source="M2ToM3",
        )
        new_id = manager.add(m3_item)
        logger.info("M2->M3: extracted lesson from %d summaries: %s", len(batch), lesson.strip()[:80])

        if processed_index:
            self.mark_done(processed_index, [item.id for item in batch])

        return new_id
