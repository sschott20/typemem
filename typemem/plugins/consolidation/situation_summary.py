"""Consolidation plugin: rolling situation summary (M1 -> single M2 item)."""

import logging
from typing import List, Optional

from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex
from typemem.plugins.base import ConsolidationPlugin

logger = logging.getLogger(__name__)


class SituationSummaryPlugin(ConsolidationPlugin):
    """Maintain a single rolling M2 summary of the current situation."""

    @property
    def name(self) -> str:
        return "situation_summary"

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
            return []

        unprocessed = self.get_unprocessed(manager, processed_index) if processed_index else manager.get_by_tier(self.source_tier)
        if not unprocessed:
            return []

        prev_items = manager.get_by_source("situation_summary", tier=MemoryTier.M2)
        previous_summary = prev_items[0].document if prev_items else "No previous summary."

        observations_text = "\n".join(f"- {item.document}" for item in unprocessed[:20])
        prompt = (
            f"Here is the previous situation summary:\n{previous_summary}\n\n"
            f"Here are new observations since then:\n{observations_text}\n\n"
            "Write an updated 2-3 sentence summary of the current situation."
        )

        try:
            summary = llm(prompt).strip()
        except Exception as e:
            logger.error("SituationSummary LLM call failed: %s", e)
            return []

        for item in prev_items:
            manager.delete(item.id)

        new_item = MemoryItem(
            document=summary,
            tier=MemoryTier.M2,
            memory_type=MemoryType.SUMMARY,
            robot_id=unprocessed[0].robot_id,
            source="situation_summary",
            keywords="situation,summary,current",
        )
        new_id = manager.add(new_item)

        if processed_index:
            self.mark_done(processed_index, [item.id for item in unprocessed])

        logger.info("SituationSummary: updated summary from %d observations", len(unprocessed))
        return [new_id]
