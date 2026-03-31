"""Consolidation plugin: M1 -> M2 (task execution patterns via LLM)."""

import logging
from typing import List, Optional

from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex
from typemem.utils import parse_summary_keywords
from typemem.plugins.base import ConsolidationPlugin
from typemem.plugins.consolidation._utils import most_common_waypoint, group_by_similarity

logger = logging.getLogger(__name__)


class TaskPatternExtractor(ConsolidationPlugin):
    """Extract task execution patterns from completed/stopped task records in M1."""

    def __init__(self, min_group_size: int = 3):
        self._min_group_size = min_group_size

    @property
    def name(self) -> str:
        return "TaskPattern"

    @property
    def source_tier(self) -> MemoryTier:
        return MemoryTier.M1

    @property
    def target_tier(self) -> MemoryTier:
        return MemoryTier.M2

    @property
    def interval_seconds(self) -> float:
        return 120.0

    def run(self, manager: MemoryManager, llm=None, processed_index: Optional[ProcessedIndex] = None) -> List[str]:
        if llm is None:
            logger.debug("TaskPattern: skipping, no LLM available")
            return []

        unprocessed = self.get_unprocessed(manager, processed_index) if processed_index else manager.get_by_tier(self.source_tier)

        # Filter to only task completion/stop records
        filtered = [
            item for item in unprocessed
            if item.memory_type == MemoryType.ACTION
            and (item.document.startswith("Task completed:") or item.document.startswith("Task stopped:"))
        ]

        if len(filtered) < self._min_group_size:
            logger.debug("TaskPattern: skipping, only %d task records (need %d)", len(filtered), self._min_group_size)
            return []

        groups = group_by_similarity(manager, filtered, MemoryTier.M1)

        new_ids = []
        for group in groups:
            if len(group) < self._min_group_size:
                continue

            text = "\n".join([f"- {item.document}" for item in group])
            prompt = (
                "Analyze these task execution records and identify patterns "
                "in how tasks were completed, including difficulties and recommendations. "
                "Then list 3-5 keywords.\n\n"
                f"{text}\n\n"
                "Summary: <execution pattern analysis>\n"
                "Keywords: <comma-separated keywords>"
            )

            try:
                response = llm(prompt)
                summary, keywords = parse_summary_keywords(response)
            except Exception as e:
                logger.error("LLM analysis failed for TaskPattern: %s", e)
                continue

            common_waypoint = most_common_waypoint(group)

            m2_item = MemoryItem(
                document=summary,
                tier=MemoryTier.M2,
                memory_type=MemoryType.SUMMARY,
                robot_id=group[0].robot_id,
                waypoint=common_waypoint,
                keywords=keywords,
                source="TaskPattern",
            )
            new_id = manager.add(m2_item)
            new_ids.append(new_id)

            if processed_index:
                self.mark_done(processed_index, [item.id for item in group])

        if new_ids:
            logger.info("TaskPattern: created %d summaries from %d task records", len(new_ids), len(filtered))
        return new_ids
