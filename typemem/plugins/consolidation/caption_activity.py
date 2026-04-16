"""Consolidation plugin: VLM captions -> activity/event detection (M1 -> M2).

Processes VLM caption observations in temporal order to identify
activities, state changes, and events in the environment.
"""

import logging
from typing import List, Optional

from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex
from typemem.utils import parse_summary_keywords
from typemem.plugins.base import ConsolidationPlugin
from typemem.plugins.consolidation._utils import most_common_waypoint

logger = logging.getLogger(__name__)

_ACTIVITY_PROMPT_TEMPLATE = (
    "These are sequential camera observations from a robot, ordered by time. "
    "Identify any activities, events, or changes that occurred between observations. "
    "Describe what changed and any ongoing activities in 1-3 sentences.\n\n"
    "{observations}\n\n"
    "Summary: <activity/event description>\n"
    "Keywords: <comma-separated keywords>"
)


class CaptionActivityConsolidator(ConsolidationPlugin):
    """Detects activities and events from sequential VLM captions."""

    def __init__(self, min_items: int = 3):
        self._min_items = min_items

    @property
    def name(self) -> str:
        return "CaptionActivity"

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
            logger.debug("CaptionActivity: skipping, no LLM available")
            return []

        unprocessed = self.get_unprocessed(manager, processed_index, tiers=[self.source_tier] if self.source_tier else None) if processed_index else manager.get_by_tier(self.source_tier)
        # Filter to VLM caption items only
        captions = [i for i in unprocessed if "vlm_caption" in i.keywords]

        if len(captions) < self._min_items:
            logger.debug("CaptionActivity: skipping, only %d captions (need %d)", len(captions), self._min_items)
            return []

        # Sort by timestamp for temporal ordering
        captions.sort(key=lambda x: x.timestamp)

        observations = "\n".join(
            f"- [{i + 1}] {item.document}" for i, item in enumerate(captions)
        )
        prompt = _ACTIVITY_PROMPT_TEMPLATE.format(observations=observations)

        try:
            response = llm(prompt)
            summary, keywords = parse_summary_keywords(response)
        except Exception as e:
            logger.error(
                "LLM activity detection failed: %s", e,
            )
            return []

        # Prepend activity and vlm_caption to keywords
        kw_parts = ["activity", "vlm_caption"]
        if keywords:
            kw_parts.extend(k for k in keywords.split(",") if k not in kw_parts)
        final_keywords = ",".join(kw_parts)

        common_waypoint = most_common_waypoint(captions)

        m2_item = MemoryItem(
            document=summary,
            tier=MemoryTier.M2,
            memory_type=MemoryType.SUMMARY,
            robot_id=captions[0].robot_id,
            waypoint=common_waypoint,
            keywords=final_keywords,
            source="CaptionActivity",
        )
        new_id = manager.add(m2_item)

        if processed_index:
            self.mark_done(processed_index, [item.id for item in captions])

        logger.info(
            "CaptionActivity: detected events from %d captions",
            len(captions),
        )
        return [new_id]
