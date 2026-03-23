"""Text-based consolidation: batch M1 observations into M2 summaries."""

from typing import List, Optional

from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.plugins.base import ConsolidationPlugin
from typemem.processed_index import ProcessedIndex


class TextSummaryPlugin(ConsolidationPlugin):
    """Concatenates unprocessed M1 items into M2 summary entries.

    Groups items into batches of ``batch_size`` and joins their text
    with semicolons.  No LLM required — purely textual concatenation.
    """

    def __init__(self, batch_size: int = 3, interval: float = 10.0):
        self._batch_size = batch_size
        self._interval = interval

    @property
    def name(self) -> str:
        return "text_summary"

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
        llm=None,
        processed_index: Optional[ProcessedIndex] = None,
    ) -> List[str]:
        unprocessed = self.get_unprocessed(manager, processed_index)
        if len(unprocessed) < self._batch_size:
            return []

        created_ids: List[str] = []
        for i in range(0, len(unprocessed), self._batch_size):
            batch = unprocessed[i : i + self._batch_size]
            if len(batch) < self._batch_size:
                break

            summary_text = "[Summary] " + "; ".join(
                item.document for item in batch
            )
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
