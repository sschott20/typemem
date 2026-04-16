import pytest
from typemem.plugins.base import ObservationPlugin, ConsolidationPlugin
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager


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
    def interval_seconds(self):
        return 60.0

    def run(self, manager, llm=None):
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
        assert con.interval_seconds == 60.0

    def test_tag_based_processed_tracking(self, manager):
        """Plugins track processed items via tags, not a separate index."""
        con = DummyConsolidator()
        item = MemoryItem(document="test", tier=MemoryTier.M1,
                          memory_type=MemoryType.OBSERVATION, robot_id="r1")
        manager.add(item)
        tag = f"processed:{con.name}"

        # Initially unprocessed
        unprocessed = manager.get_by_tag(tag, exclude=True)
        assert len(unprocessed) == 1

        # After tagging, it's filtered out
        manager.add_tag(item.id, tag)
        unprocessed = manager.get_by_tag(tag, exclude=True)
        assert len(unprocessed) == 0

        # Reverse filter finds the tagged items
        processed = manager.get_by_tag(tag)
        assert len(processed) == 1
        assert processed[0].id == item.id
