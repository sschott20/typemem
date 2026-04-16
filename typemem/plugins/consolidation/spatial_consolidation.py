"""Consolidation plugin: M1 -> M2 (group observations by location/waypoint via LLM)."""

import logging
from typing import Dict, List, Optional

from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex
from typemem.utils import parse_summary_keywords
from typemem.plugins.base import ConsolidationPlugin

logger = logging.getLogger(__name__)


class SpatialConsolidation(ConsolidationPlugin):
    """Group M1 observations by waypoint and summarize each location via LLM."""

    def __init__(self, min_group_size: int = 3):
        self._min_group_size = min_group_size

    @property
    def name(self) -> str:
        return "SpatialConsolidation"

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
            logger.debug("SpatialConsolidation: skipping, no LLM available")
            return []

        unprocessed = self.get_unprocessed(manager, processed_index, tiers=[self.source_tier] if self.source_tier else None) if processed_index else manager.get_by_tier(self.source_tier)

        # Group by exact waypoint value, discard items where waypoint is None
        waypoint_groups: Dict[int, List[MemoryItem]] = {}
        for item in unprocessed:
            if item.waypoint is None:
                continue
            waypoint_groups.setdefault(item.waypoint, []).append(item)

        new_ids = []
        for waypoint, group in waypoint_groups.items():
            if len(group) < self._min_group_size:
                continue

            observations_text = "\n".join([f"- {item.document}" for item in group])
            prompt = (
                f"Summarize these observations from the same location (waypoint {waypoint}) "
                "into a concise location profile, then list 3-5 keywords.\n\n"
                f"{observations_text}\n\n"
                "Summary: <location summary>\n"
                "Keywords: <comma-separated keywords>"
            )

            try:
                response = llm(prompt)
                summary, keywords = parse_summary_keywords(response)
            except Exception as e:
                logger.error(
                    "LLM summarization failed for SpatialConsolidation at waypoint %s: %s",
                    waypoint, e,
                )
                continue

            m2_item = MemoryItem(
                document=summary,
                tier=MemoryTier.M2,
                memory_type=MemoryType.SUMMARY,
                robot_id=group[0].robot_id,
                waypoint=waypoint,
                keywords=keywords,
                source="SpatialConsolidation",
            )
            new_id = manager.add(m2_item)
            new_ids.append(new_id)

            if processed_index:
                self.mark_done(processed_index, [item.id for item in group])

        if new_ids:
            logger.info(
                "SpatialConsolidation: created %d location summaries from %d observations",
                len(new_ids), len(unprocessed),
            )
        return new_ids
