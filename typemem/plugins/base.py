"""Base classes for memory observation, consolidation, and injection plugins."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from typemem.memory_item import MemoryTier
from typemem.memory_manager import MemoryManager

if TYPE_CHECKING:
    from typemem.memory_item import MemoryItem
    from typemem.plugins.runner import ObservationRunner


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


class InjectionPlugin(ABC):
    """Plugin that builds the prompt context string for a specific planning stage.

    Each stage (e.g. S1, S2, S3) gets its own InjectionPlugin. The plugin has
    full control over how memory items are retrieved, filtered, scored, and
    formatted. This enables domain-specific retrieval strategies per stage
    (e.g. task-type-specific queries, multi-pass retrieval, cross-referenced
    filtering) that a pure config-driven injector cannot express.

    Lifecycle:
        1. __init__() — construct with any plugin-specific options
        2. setup()    — called once at startup with runtime dependencies
        3. build_context(query, ctx) — called per plan call to produce prompt text
    """

    @property
    @abstractmethod
    def stage(self) -> str:
        """Stage identifier (e.g. 'S1', 'S2', 'S3'). One plugin per stage."""
        ...

    def setup(
        self,
        memory_manager: MemoryManager,
        runner: "ObservationRunner",
    ) -> None:
        """Called once at startup. Store manager + runner for later use."""
        self._manager = memory_manager
        self._runner = runner

    @abstractmethod
    def build_context(self, query: str, ctx: Optional[Dict[str, Any]] = None) -> str:
        """Build the memory context string for this stage's prompt.

        Args:
            query: The query string (typically the task content).
            ctx: Optional dict of extra context from the caller (e.g. pcb, subtask).
                 Plugins can ignore this or use it for specialized queries.

        Returns:
            The formatted memory context to inject into the stage's prompt.
        """
        ...

    def last_results(self) -> List[Dict[str, Any]]:
        """Return details of items included in the most recent build_context call.
        Used by logging / viz. Each dict has: id, text, tier, score, source."""
        return getattr(self, "_last_results", [])


class ConsolidationPlugin(ABC):
    """Plugin that reads memory and produces memory or side effects.

    Plugins are arbitrary — input is some memories, output is some memories
    (or deletions, or events pushed to the bus). The framework does not
    prescribe which tiers a plugin reads from or writes to, nor does it
    track which items a plugin has processed. Plugins use tags on memory
    items to communicate state (e.g. add "processed:myplugin" after handling).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def interval_seconds(self) -> float:
        ...

    @abstractmethod
    def run(
        self,
        manager: MemoryManager,
        llm=None,
    ) -> List[str]:
        """Run the plugin. Return list of newly-created item IDs."""
        ...
