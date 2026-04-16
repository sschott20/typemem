"""Consolidation plugin: M1 -> M2 (action patterns via LLM)."""

import logging
import re
from collections import defaultdict
from typing import List, Optional

from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex
from typemem.utils import parse_summary_keywords
from typemem.plugins.base import ConsolidationPlugin

logger = logging.getLogger(__name__)

# Matches observer format: "Action: walk(forward) | Result: SUCCESS ..."
_ACTION_PATTERN = re.compile(r'Action:\s*(\S+)')


class ActionPatternExtractor(ConsolidationPlugin):
    """Extract success/failure patterns from groups of M1 ACTION items."""

    def __init__(self, min_group_size: int = 3):
        self._min_group_size = min_group_size

    @property
    def name(self) -> str:
        return "ActionPattern"

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
            logger.debug("ActionPattern: skipping, no LLM available")
            return []

        unprocessed = self.get_unprocessed(manager, processed_index, tiers=[self.source_tier] if self.source_tier else None) if processed_index else manager.get_by_tier(self.source_tier)

        # Filter to ACTION items whose document matches the action pattern
        groups: dict[str, list[MemoryItem]] = defaultdict(list)
        for item in unprocessed:
            if item.memory_type != MemoryType.ACTION:
                continue
            match = _ACTION_PATTERN.search(item.document)
            if not match:
                continue
            skill_name = match.group(1)
            groups[skill_name].append(item)

        new_ids: List[str] = []
        for skill_name, group in groups.items():
            if len(group) < self._min_group_size:
                continue

            observations_text = "\n".join([f"- {item.document}" for item in group])
            prompt = (
                "Analyze these robot action outcomes and identify patterns of success/failure "
                "for the skill, then list 3-5 keywords.\n\n"
                f"{observations_text}\n\n"
                "Summary: <pattern analysis>\n"
                "Keywords: <comma-separated keywords>"
            )

            try:
                response = llm(prompt)
                summary, keywords = parse_summary_keywords(response)
            except Exception as e:
                logger.error(
                    "LLM action pattern analysis failed for skill '%s': %s",
                    skill_name, e,
                )
                continue

            m2_item = MemoryItem(
                document=summary,
                tier=MemoryTier.M2,
                memory_type=MemoryType.SUMMARY,
                robot_id=group[0].robot_id,
                keywords=keywords,
                source="ActionPattern",
            )
            new_id = manager.add(m2_item)
            new_ids.append(new_id)

            if processed_index:
                self.mark_done(processed_index, [item.id for item in group])

        if new_ids:
            logger.info(
                "ActionPattern: created %d pattern summaries",
                len(new_ids),
            )
        return new_ids
