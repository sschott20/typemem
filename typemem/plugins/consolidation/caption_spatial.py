"""Consolidation plugin: VLM captions -> spatial knowledge (M1 -> M2).

Groups VLM caption observations by waypoint and summarizes into
persistent spatial descriptions of the environment layout.
"""

import logging
from typing import List, Optional

from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex
from typemem.utils import parse_summary_keywords
from typemem.plugins.base import ConsolidationPlugin

logger = logging.getLogger(__name__)

_SPATIAL_PROMPT_TEMPLATE = (
    "Based on these camera observations from the robot{location_ctx}, "
    "describe the spatial layout of the environment in 1-3 sentences. "
    "Focus on: what objects are present, their positions relative to each other, "
    "and key landmarks or features.\n\n"
    "{observations}\n\n"
    "Summary: <spatial description>\n"
    "Keywords: <comma-separated keywords>"
)


class CaptionSpatialConsolidator(ConsolidationPlugin):
    """Groups VLM caption M1 items by waypoint and creates M2 spatial summaries."""

    def __init__(self, min_group_size: int = 3):
        self._min_group_size = min_group_size

    @property
    def name(self) -> str:
        return "CaptionSpatial"

    @property
    def source_tier(self) -> MemoryTier:
        return MemoryTier.M1

    @property
    def target_tier(self) -> MemoryTier:
        return MemoryTier.M2

    @property
    def interval_seconds(self) -> float:
        return 60.0

    def run(self, manager: MemoryManager, llm=None, processed_index: Optional[ProcessedIndex] = None) -> List[str]:
        if llm is None:
            logger.debug("CaptionSpatial: skipping, no LLM available")
            return []

        unprocessed = self.get_unprocessed(manager, processed_index) if processed_index else manager.get_by_tier(self.source_tier)
        # Filter to VLM caption items only
        captions = [i for i in unprocessed if "vlm_caption" in i.keywords]

        if len(captions) < self._min_group_size:
            logger.debug("CaptionSpatial: skipping, only %d captions (need %d)", len(captions), self._min_group_size)
            return []

        groups = self._group_by_waypoint(captions)

        new_ids = []
        for waypoint, group in groups.items():
            if len(group) < self._min_group_size:
                continue

            location_ctx = f" at waypoint {waypoint}" if waypoint is not None else ""
            observations = "\n".join(f"- {item.document}" for item in group)
            prompt = _SPATIAL_PROMPT_TEMPLATE.format(
                location_ctx=location_ctx,
                observations=observations,
            )

            try:
                response = llm(prompt)
                summary, keywords = parse_summary_keywords(response)
            except Exception as e:
                logger.error(
                    "LLM spatial summarization failed: %s", e,
                )
                continue

            # Prepend spatial and vlm_caption to keywords
            kw_parts = ["spatial", "vlm_caption"]
            if keywords:
                kw_parts.extend(k for k in keywords.split(",") if k not in kw_parts)
            final_keywords = ",".join(kw_parts)

            m2_item = MemoryItem(
                document=summary,
                tier=MemoryTier.M2,
                memory_type=MemoryType.SUMMARY,
                robot_id=group[0].robot_id,
                waypoint=waypoint,
                keywords=final_keywords,
                source="CaptionSpatial",
            )
            new_id = manager.add(m2_item)
            new_ids.append(new_id)

            if processed_index:
                self.mark_done(processed_index, [item.id for item in group])

        if new_ids:
            logger.info(
                "CaptionSpatial: created %d spatial summaries",
                len(new_ids),
            )
        return new_ids

    def _group_by_waypoint(self, items: List[MemoryItem]) -> dict:
        """Group items by waypoint. Items with no waypoint go into None key."""
        groups: dict = {}
        for item in items:
            wp = item.waypoint
            groups.setdefault(wp, []).append(item)
        return groups
