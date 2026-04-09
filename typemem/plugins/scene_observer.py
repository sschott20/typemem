"""Observation plugin: turn raw ``scene_object`` events into M1 memory items.

Subscribes to the ``scene_object`` channel on the typemem event bus, deduplicates
detections by ``(name, waypoint)`` in-process, and writes one M1 ``MemoryItem``
per unique pair as ``"Saw <name> at waypoint <wp>"``.

In-process dedup is essential when consuming real recordings: a 2-minute robot
session can emit thousands of redundant ``scene_object`` events for the same
object, and routing all of them through ``MemoryManager.add()`` would pay the
ChromaDB embedding cost on every event before manager-level dedup runs. Filtering
client-side keeps replay throughput practical.

Only ``scene_object`` is handled today. Other recorded channels (``action``,
``user_instruction``, ``system_response``, ``task_lifecycle``) are TODO follow-ups.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from typemem import events
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.plugins.base import ObservationPlugin


_CHANNEL = "scene_object"
_DEFAULT_QUEUE_SIZE = 100_000


class SceneObserver(ObservationPlugin):
    """Subscribes to ``scene_object`` and emits M1 items for new (name, waypoint) pairs."""

    def __init__(self, interval_seconds: float = 1.0, queue_size: int = _DEFAULT_QUEUE_SIZE):
        self._interval = interval_seconds
        self._queue_size = queue_size
        self._dq = None
        self._manager: Optional[MemoryManager] = None
        self._robot_id: str = ""
        self._seen: Set[Tuple[str, int]] = set()

    @property
    def name(self) -> str:
        return "scene_observer"

    @property
    def interval_seconds(self) -> float:
        return self._interval

    def setup(self, memory_manager: MemoryManager, robot_id: str) -> None:
        self._manager = memory_manager
        self._robot_id = robot_id
        self._dq = events.subscribe(_CHANNEL, maxlen=self._queue_size)

    def run(self) -> List[str]:
        if self._manager is None or self._dq is None:
            return []

        pending = events.drain(self._dq)
        emitted: List[str] = []

        for data in pending:
            name = data.get("name")
            waypoint = data.get("waypoint")
            if name is None or waypoint is None:
                continue

            key = (name, int(waypoint))
            if key in self._seen:
                continue
            self._seen.add(key)

            item = MemoryItem(
                document=f"Saw {name} at waypoint {waypoint}",
                tier=MemoryTier.M1,
                memory_type=MemoryType.OBSERVATION,
                robot_id=self._robot_id,
                waypoint=int(waypoint),
                source=self.name,
            )
            # SceneObserver already dedupes structurally by (name, waypoint),
            # which is more precise than embedding-distance dedup. The
            # embedder treats "Saw chair at waypoint 1" and "Saw chair at
            # waypoint 2" as near-identical, so leaving manager dedup on
            # would silently collapse meaningfully distinct sightings.
            self._manager.add(item, skip_dedup=True)
            emitted.append(item.id)

        return emitted

    def teardown(self) -> None:
        # Subscribed deque is left in the registry; benchmark/test fixtures
        # call events.reset() between runs.
        self._dq = None
        self._seen.clear()
