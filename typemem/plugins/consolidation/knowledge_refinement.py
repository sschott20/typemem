"""Consolidation plugin: M3 -> M3 (deduplicate/merge similar long-term knowledge via LLM)."""

import logging
from typing import List, Optional

from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex
from typemem.utils import parse_summary_keywords
from typemem.plugins.base import ConsolidationPlugin
from typemem.plugins.consolidation._utils import group_by_similarity

logger = logging.getLogger(__name__)


class KnowledgeRefinement(ConsolidationPlugin):
    """Deduplicate and merge similar M3 long-term knowledge via LLM."""

    def __init__(self, similarity_threshold: float = 0.15):
        self._similarity_threshold = similarity_threshold

    @property
    def name(self) -> str:
        return "KnowledgeRefinement"

    @property
    def source_tier(self) -> MemoryTier:
        return MemoryTier.M3

    @property
    def target_tier(self) -> MemoryTier:
        return MemoryTier.M3

    @property
    def interval_seconds(self) -> float:
        return 1800.0

    def run(self, manager: MemoryManager, llm=None, processed_index: Optional[ProcessedIndex] = None) -> List[str]:
        if llm is None:
            logger.debug("KnowledgeRefinement: skipping, no LLM available")
            return []

        unprocessed = self.get_unprocessed(manager, processed_index, tiers=[self.source_tier] if self.source_tier else None) if processed_index else manager.get_by_tier(self.source_tier)

        if len(unprocessed) < 2:
            logger.debug("KnowledgeRefinement: skipping, only %d unprocessed M3 items (need 2)", len(unprocessed))
            return []

        groups = group_by_similarity(manager, unprocessed, MemoryTier.M3, distance_threshold=self._similarity_threshold)

        new_ids = []
        for group in groups:
            if len(group) < 2:
                continue
            text = "\n".join([f"- {item.document}" for item in group])
            prompt = (
                "These long-term robot memories are very similar. "
                "If they overlap, merge them into a single refined lesson with keywords. "
                "If they are actually distinct, respond exactly: KEEP_ALL\n\n"
                f"{text}\n\n"
                "Lesson: <merged lesson or KEEP_ALL>\n"
                "Keywords: <comma-separated keywords>"
            )

            try:
                response = llm(prompt)
            except Exception as e:
                logger.error(
                    "LLM merge failed for KnowledgeRefinement: %s", e,
                )
                continue

            if "KEEP_ALL" in response.upper():
                continue

            try:
                lesson, keywords = parse_summary_keywords(response)
            except Exception as e:
                logger.error(
                    "Failed to parse KnowledgeRefinement response: %s", e,
                )
                continue

            m3_item = MemoryItem(
                document=lesson,
                tier=MemoryTier.M3,
                memory_type=MemoryType.LESSON,
                robot_id=group[0].robot_id,
                keywords=keywords,
                source="KnowledgeRefinement",
            )
            new_id = manager.add(m3_item)
            new_ids.append(new_id)

            if processed_index:
                self.mark_done(processed_index, [item.id for item in group])

        if new_ids:
            logger.info(
                "KnowledgeRefinement: merged %d groups from M3",
                len(new_ids),
            )
        return new_ids
