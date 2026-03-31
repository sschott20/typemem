"""End-to-end integration test: event bus -> observation -> consolidation -> injection."""

import time
import pytest
from typemem import (
    create_memory_system, events,
    MemoryItem, MemoryTier, MemoryType,
)
from typemem.plugins.base import ObservationPlugin, ConsolidationPlugin
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex


class EventDrivenObserver(ObservationPlugin):
    """Test observer that subscribes to 'test_sensor' channel."""

    def __init__(self):
        self._deque = None
        self._manager = None
        self._robot_id = ""

    @property
    def name(self):
        return "event_observer"

    @property
    def interval_seconds(self):
        return 0.1

    def setup(self, memory_manager, robot_id):
        self._manager = memory_manager
        self._robot_id = robot_id
        self._deque = events.subscribe("test_sensor")

    def run(self):
        created = []
        while self._deque:
            event = self._deque.popleft()
            item = MemoryItem(
                document=event["text"],
                tier=MemoryTier.M1,
                memory_type=MemoryType.OBSERVATION,
                robot_id=self._robot_id,
            )
            mid = self._manager.add(item)
            created.append(mid)
        return created


class SimpleM1toM2(ConsolidationPlugin):
    @property
    def name(self):
        return "m1_to_m2"

    @property
    def source_tier(self):
        return MemoryTier.M1

    @property
    def target_tier(self):
        return MemoryTier.M2

    @property
    def interval_seconds(self):
        return 1.0

    def run(self, manager, llm=None, processed_index=None):
        unprocessed = self.get_unprocessed(manager, processed_index)
        if len(unprocessed) < 3:
            return []
        summary = "[Summary] " + "; ".join(i.document for i in unprocessed)
        item = MemoryItem(
            document=summary, tier=MemoryTier.M2,
            memory_type=MemoryType.SUMMARY, robot_id="test",
        )
        mid = manager.add(item)
        self.mark_done(processed_index, [i.id for i in unprocessed])
        return [mid]


@pytest.fixture(autouse=True)
def reset_events():
    events.reset()
    yield
    events.reset()


class TestIntegration:
    def test_full_pipeline(self, tmp_path):
        """Event bus -> observer -> store -> consolidation -> injection."""
        manager, engine, injector, recorder, runner, fs = create_memory_system(
            persist_dir=str(tmp_path / "chroma"),
            robot_id="test_robot",
            plugins=[EventDrivenObserver(), SimpleM1toM2()],
        )

        # Start observation runner
        runner.start(manager, "test_robot", tick_interval=0.05)

        # Push events through event bus
        for i in range(5):
            events.push("test_sensor", {"text": f"saw object {i} on table"})
            time.sleep(0.02)

        # Wait for observer to drain events
        time.sleep(0.5)
        runner.stop()

        # Verify observations were stored
        m1_count = manager.count(tier=MemoryTier.M1)
        assert m1_count >= 3, f"Expected >= 3 M1 items, got {m1_count}"

        # Run consolidation
        results = engine.run_all()
        assert "m1_to_m2" in results
        assert len(results["m1_to_m2"]) >= 1

        # Verify M2 summary created
        assert manager.count(tier=MemoryTier.M2) >= 1

        # Inject context
        context = injector.inject("S1", "what objects are on the table")
        assert context != ""
        assert "object" in context.lower()

    def test_frame_store_integration(self, tmp_path):
        import numpy as np
        manager, engine, injector, recorder, runner, fs = create_memory_system(
            persist_dir=str(tmp_path / "chroma"),
            robot_id="r1",
            frame_store_dir=str(tmp_path / "frames"),
        )
        assert fs is not None

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        fid = fs.store(frame, timestamp=time.time())

        # Store a memory referencing the frame
        item = MemoryItem(
            document="saw a frame", tier=MemoryTier.M1,
            memory_type=MemoryType.OBSERVATION, robot_id="r1",
        )
        manager.add(item)
        loaded = fs.load(fid)
        assert loaded is not None

    def test_create_with_explicit_plugin_list(self, tmp_path):
        from typemem.plugins.consolidation import M1ToM2Strategy
        from typemem.plugins.text_summary import TextSummaryPlugin

        plugins = [TextSummaryPlugin(batch_size=3, interval=0), M1ToM2Strategy()]

        manager, engine, injector, recorder, obs_runner, frame_store = create_memory_system(
            persist_dir=str(tmp_path / "chroma"),
            robot_id="test",
            plugins=plugins,
        )
        assert "M1ToM2" in engine.list_strategies()
        assert manager is not None

    def test_create_with_no_plugins(self, tmp_path):
        manager, engine, injector, recorder, obs_runner, frame_store = create_memory_system(
            persist_dir=str(tmp_path / "chroma"),
            robot_id="test",
        )
        assert engine.list_strategies() == []
        assert obs_runner.list_plugins() == []
