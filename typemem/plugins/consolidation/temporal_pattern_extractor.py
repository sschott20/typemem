"""Consolidation plugin: M2 -> M3 (extract temporal patterns from summaries via LLM)."""

import datetime
import logging
from typing import Dict, List, Optional

from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex
from typemem.utils import parse_summary_keywords
from typemem.plugins.base import ConsolidationPlugin

logger = logging.getLogger(__name__)


class TemporalPatternExtractor(ConsolidationPlugin):
    """Extract temporal patterns from M2 summaries grouped by time-of-day."""

    def __init__(self, min_summaries: int = 3, min_bin_size: int = 2):
        self._min_summaries = min_summaries
        self._min_bin_size = min_bin_size

    @property
    def name(self) -> str:
        return "TemporalPattern"

    @property
    def source_tier(self) -> MemoryTier:
        return MemoryTier.M2

    @property
    def target_tier(self) -> MemoryTier:
        return MemoryTier.M3

    @property
    def interval_seconds(self) -> float:
        return 600.0

    def run(self, manager: MemoryManager, llm=None, processed_index: Optional[ProcessedIndex] = None) -> List[str]:
        if llm is None:
            logger.debug("TemporalPattern: skipping, no LLM available")
            return []

        unprocessed = self.get_unprocessed(manager, processed_index, tiers=[self.source_tier] if self.source_tier else None) if processed_index else manager.get_by_tier(self.source_tier)

        if len(unprocessed) < self._min_summaries:
            logger.debug("TemporalPattern: skipping, only %d summaries (need %d)", len(unprocessed), self._min_summaries)
            return []

        # Bin items by time-of-day period
        bins: Dict[str, List[MemoryItem]] = {}
        for item in unprocessed:
            hour = datetime.datetime.fromtimestamp(item.timestamp).hour
            if 6 <= hour < 12:
                period = "morning"
            elif 12 <= hour < 18:
                period = "afternoon"
            elif 18 <= hour < 22:
                period = "evening"
            else:
                period = "night"
            bins.setdefault(period, []).append(item)

        new_ids: List[str] = []
        for period, items in bins.items():
            if len(items) < self._min_bin_size:
                continue

            text = "\n".join([f"- {item.document}" for item in items])
            prompt = (
                f"These observations all occurred during the {period} period. "
                "Extract a temporal pattern or lesson about what typically happens "
                "during this time, then list 3-5 keywords.\n\n"
                f"{text}\n\n"
                "Lesson: <temporal pattern>\n"
                "Keywords: <comma-separated keywords>"
            )

            try:
                response = llm(prompt)
                lesson, keywords = parse_summary_keywords(response)
            except Exception as e:
                logger.error(
                    "LLM temporal pattern extraction failed for %s: %s",
                    period, e,
                )
                continue

            m3_item = MemoryItem(
                document=lesson,
                tier=MemoryTier.M3,
                memory_type=MemoryType.LESSON,
                robot_id=items[0].robot_id,
                keywords=keywords,
                source="TemporalPattern",
            )
            new_id = manager.add(m3_item)
            new_ids.append(new_id)

            if processed_index:
                self.mark_done(processed_index, [item.id for item in items])

        if new_ids:
            logger.info(
                "TemporalPattern: created %d temporal lessons from %d summaries",
                len(new_ids), len(unprocessed),
            )
        return new_ids
