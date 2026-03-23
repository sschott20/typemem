"""Tests for the TextSummaryPlugin consolidation plugin."""

import pytest
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex
from typemem.plugins.text_summary import TextSummaryPlugin


@pytest.fixture
def manager(tmp_path):
    return MemoryManager(persist_dir=str(tmp_path / "chroma"), robot_id="test")


@pytest.fixture
def processed_index(tmp_path):
    return ProcessedIndex(str(tmp_path / "processed.json"))


def _add_m1(manager, text, robot_id="test"):
    item = MemoryItem(
        document=text, tier=MemoryTier.M1,
        memory_type=MemoryType.OBSERVATION, robot_id=robot_id,
    )
    return manager.add(item)


class TestTextSummaryPlugin:
    def test_properties(self):
        plugin = TextSummaryPlugin(batch_size=5, interval=30.0)
        assert plugin.name == "text_summary"
        assert plugin.source_tier == MemoryTier.M1
        assert plugin.target_tier == MemoryTier.M2
        assert plugin.interval_seconds == 30.0

    def test_no_consolidation_below_batch_size(self, manager, processed_index):
        plugin = TextSummaryPlugin(batch_size=3)
        _add_m1(manager, "obs 1")
        _add_m1(manager, "obs 2")

        result = plugin.run(manager, processed_index=processed_index)
        assert result == []
        assert manager.count(tier=MemoryTier.M2) == 0

    def test_consolidates_at_batch_size(self, manager, processed_index):
        plugin = TextSummaryPlugin(batch_size=3)
        _add_m1(manager, "saw a cup")
        _add_m1(manager, "saw a plate")
        _add_m1(manager, "saw a person")

        result = plugin.run(manager, processed_index=processed_index)
        assert len(result) == 1
        assert manager.count(tier=MemoryTier.M2) == 1

        m2_items = manager.get_by_tier(MemoryTier.M2)
        assert len(m2_items) == 1
        assert "[Summary]" in m2_items[0].document
        assert "cup" in m2_items[0].document
        assert "plate" in m2_items[0].document
        assert "person" in m2_items[0].document

    def test_multiple_batches(self, manager, processed_index):
        plugin = TextSummaryPlugin(batch_size=2)
        for i in range(5):
            _add_m1(manager, f"observation {i}")

        result = plugin.run(manager, processed_index=processed_index)
        # 5 items, batch_size=2 → 2 full batches, 1 leftover
        assert len(result) == 2
        assert manager.count(tier=MemoryTier.M2) == 2

    def test_idempotent_run(self, manager, processed_index):
        plugin = TextSummaryPlugin(batch_size=3)
        _add_m1(manager, "red cup on the kitchen counter")
        _add_m1(manager, "blue plate on dining table")
        _add_m1(manager, "person entered from hallway")

        result1 = plugin.run(manager, processed_index=processed_index)
        assert len(result1) == 1

        # Second run should produce nothing (items already processed)
        result2 = plugin.run(manager, processed_index=processed_index)
        assert result2 == []

    def test_new_items_after_consolidation(self, manager, processed_index):
        plugin = TextSummaryPlugin(batch_size=2)
        _add_m1(manager, "first")
        _add_m1(manager, "second")

        result1 = plugin.run(manager, processed_index=processed_index)
        assert len(result1) == 1

        # Add more items
        _add_m1(manager, "third")
        _add_m1(manager, "fourth")

        result2 = plugin.run(manager, processed_index=processed_index)
        assert len(result2) == 1
        assert manager.count(tier=MemoryTier.M2) == 2

    def test_preserves_robot_id(self, manager, processed_index):
        plugin = TextSummaryPlugin(batch_size=2)
        _add_m1(manager, "obs 1", robot_id="test")
        _add_m1(manager, "obs 2", robot_id="test")

        plugin.run(manager, processed_index=processed_index)
        m2_items = manager.get_by_tier(MemoryTier.M2)
        assert m2_items[0].robot_id == "test"

    def test_default_batch_size(self):
        plugin = TextSummaryPlugin()
        assert plugin._batch_size == 3
        assert plugin._interval == 10.0
