import pytest
from typemem.plugins.base import ObservationPlugin, ConsolidationPlugin
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex


class DummyObserver(ObservationPlugin):
    @property
    def name(self):
        return "dummy_observer"

    @property
    def interval_seconds(self):
        return 1.0

    def run(self):
        return []


class DummyConsolidator(ConsolidationPlugin):
    @property
    def name(self):
        return "dummy_consolidator"

    @property
    def source_tier(self):
        return MemoryTier.M1

    @property
    def target_tier(self):
        return MemoryTier.M2

    @property
    def interval_seconds(self):
        return 60.0

    def run(self, manager, llm=None, processed_index=None):
        return []


class TestObservationPlugin:
    def test_instantiate(self):
        obs = DummyObserver()
        assert obs.name == "dummy_observer"
        assert obs.interval_seconds == 1.0

    def test_setup_default_noop(self, manager):
        obs = DummyObserver()
        obs.setup(manager, "robot1")

    def test_teardown_default_noop(self):
        obs = DummyObserver()
        obs.teardown()

    def test_run_returns_list(self):
        obs = DummyObserver()
        assert obs.run() == []


class TestConsolidationPlugin:
    def test_instantiate(self):
        con = DummyConsolidator()
        assert con.name == "dummy_consolidator"
        assert con.source_tier == MemoryTier.M1
        assert con.target_tier == MemoryTier.M2

    def test_get_unprocessed(self, manager, tmp_path):
        con = DummyConsolidator()
        idx = ProcessedIndex(str(tmp_path / "proc.json"))
        item = MemoryItem(document="test", tier=MemoryTier.M1,
                          memory_type=MemoryType.OBSERVATION, robot_id="r1")
        manager.add(item)
        unprocessed = con.get_unprocessed(manager, idx)
        assert len(unprocessed) == 1
        assert unprocessed[0].id == item.id

    def test_mark_done(self, tmp_path):
        con = DummyConsolidator()
        idx = ProcessedIndex(str(tmp_path / "proc.json"))
        con.mark_done(idx, ["id1", "id2"])
        assert idx.is_processed("dummy_consolidator", "id1")
        assert idx.is_processed("dummy_consolidator", "id2")

    def test_get_unprocessed_filters_processed(self, manager, tmp_path):
        con = DummyConsolidator()
        idx = ProcessedIndex(str(tmp_path / "proc.json"))
        item1 = MemoryItem(document="first", tier=MemoryTier.M1,
                           memory_type=MemoryType.OBSERVATION, robot_id="r1")
        item2 = MemoryItem(document="second", tier=MemoryTier.M1,
                           memory_type=MemoryType.OBSERVATION, robot_id="r1")
        manager.add(item1)
        manager.add(item2)
        con.mark_done(idx, [item1.id])
        unprocessed = con.get_unprocessed(manager, idx)
        assert len(unprocessed) == 1
        assert unprocessed[0].id == item2.id
