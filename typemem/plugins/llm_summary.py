"""LLM-based consolidation: summarize M1 observations into M2 using an LLM."""

import logging
from typing import List, Optional

from typemem.llm import LLMCallable
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.plugins.base import ConsolidationPlugin
from typemem.processed_index import ProcessedIndex

logger = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = """Summarize these robot observations into a single concise memory entry. \
Keep only the key facts — what was seen, where, and any changes. Be brief.

Observations:
{observations}

Summary:"""


class LLMSummaryPlugin(ConsolidationPlugin):
    """Summarizes unprocessed M1 items into M2 entries using an LLM.

    Falls back to doing nothing if no LLM callable is provided.
    """

    def __init__(self, batch_size: int = 3, interval: float = 10.0):
        self._batch_size = batch_size
        self._interval = interval

    @property
    def name(self) -> str:
        return "llm_summary"

    @property
    def source_tier(self) -> MemoryTier:
        return MemoryTier.M1

    @property
    def target_tier(self) -> MemoryTier:
        return MemoryTier.M2

    @property
    def interval_seconds(self) -> float:
        return self._interval

    def run(
        self,
        manager: MemoryManager,
        llm: Optional[LLMCallable] = None,
        processed_index: Optional[ProcessedIndex] = None,
    ) -> List[str]:
        if llm is None:
            return []

        unprocessed = self.get_unprocessed(manager, processed_index)
        if len(unprocessed) < self._batch_size:
            return []

        created_ids: List[str] = []
        for i in range(0, len(unprocessed), self._batch_size):
            batch = unprocessed[i : i + self._batch_size]
            if len(batch) < self._batch_size:
                break

            observations = "\n".join(
                f"- {item.document}" for item in batch
            )
            prompt = _SUMMARIZE_PROMPT.format(observations=observations)

            try:
                summary_text = llm(prompt)
            except Exception as e:
                logger.error("LLM call failed during consolidation: %s", e)
                continue

            robot_id = batch[0].robot_id
            summary_item = MemoryItem(
                document=summary_text,
                tier=MemoryTier.M2,
                memory_type=MemoryType.SUMMARY,
                robot_id=robot_id,
            )
            mid = manager.add(summary_item)
            created_ids.append(mid)
            self.mark_done(processed_index, [item.id for item in batch])

        return created_ids
