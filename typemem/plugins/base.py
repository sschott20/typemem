"""Base classes for memory observation and consolidation plugins."""

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

from typemem.memory_item import MemoryTier
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex

if TYPE_CHECKING:
    from typemem.memory_item import MemoryItem


class ObservationPlugin(ABC):
    """Plugin that reads data (via event bus) and writes items to memory.

    Lifecycle:
        1. __init__() — no-arg constructor for auto-discovery
        2. setup()    — called once at startup with runtime dependencies
        3. run()      — called on timer at interval_seconds
        4. teardown() — called on shutdown
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def interval_seconds(self) -> float:
        ...

    def setup(self, memory_manager: MemoryManager, robot_id: str) -> None:
        pass

    @abstractmethod
    def run(self) -> List[str]:
        ...

    def live_summary(self) -> Optional[str]:
        """Override to expose current state for prompt injection.
        Called by the injector when this plugin's name appears in a stage's live_sources list.
        Must be fast (no LLM calls). Return None if no data available."""
        return None

    def teardown(self) -> None:
        pass


class ConsolidationPlugin(ABC):
    """Plugin that transforms memories between tiers."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def source_tier(self) -> MemoryTier:
        ...

    @property
    @abstractmethod
    def target_tier(self) -> MemoryTier:
        ...

    @property
    @abstractmethod
    def interval_seconds(self) -> float:
        ...

    def get_unprocessed(
        self,
        manager: MemoryManager,
        processed_index: ProcessedIndex,
    ) -> List["MemoryItem"]:
        all_items = manager.get_by_tier(self.source_tier)
        unprocessed_ids = set(processed_index.filter_unprocessed(
            self.name, [item.id for item in all_items],
        ))
        return [item for item in all_items if item.id in unprocessed_ids]

    def mark_done(
        self,
        processed_index: ProcessedIndex,
        item_ids: List[str],
    ) -> None:
        processed_index.mark_processed(self.name, item_ids)

    @abstractmethod
    def run(
        self,
        manager: MemoryManager,
        llm=None,
        processed_index: Optional[ProcessedIndex] = None,
    ) -> List[str]:
        ...
