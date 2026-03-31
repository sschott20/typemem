"""Tests for the SpatialConsolidation plugin."""

import pytest
from unittest.mock import MagicMock

from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex
from typemem.plugins.consolidation.spatial_consolidation import SpatialConsolidation


class TestSpatialConsolidation:

    @pytest.fixture
    def processed_index(self, tmp_path):
        return ProcessedIndex(str(tmp_path / "processed.json"))

    @pytest.fixture
    def manager(self, tmp_path):
        mgr = MemoryManager(persist_dir=str(tmp_path / "chroma"), robot_id="go2_test")

        # 4 items at waypoint=5 (kitchen-related)
        for text in [
            "There is a wooden table in the center of the kitchen",
            "I see two chairs near the kitchen counter",
            "The kitchen has a refrigerator on the left wall",
            "A microwave is placed on the kitchen shelf",
        ]:
            mgr.add(
                MemoryItem(
                    document=text,
                    tier=MemoryTier.M1,
                    memory_type=MemoryType.OBSERVATION,
                    robot_id="go2_test",
                    waypoint=5,
                ),
                auto_link=False,
            )

        # 3 items at waypoint=10 (hallway-related)
        for text in [
            "The hallway has a long carpet runner",
            "I see a coat rack near the hallway entrance",
            "There is a mirror mounted on the hallway wall",
        ]:
            mgr.add(
                MemoryItem(
                    document=text,
                    tier=MemoryTier.M1,
                    memory_type=MemoryType.OBSERVATION,
                    robot_id="go2_test",
                    waypoint=10,
                ),
                auto_link=False,
            )

        # 2 items with waypoint=None (should be ignored)
        for text in [
            "Battery level is at 80 percent",
            "System temperature is normal",
        ]:
            mgr.add(
                MemoryItem(
                    document=text,
                    tier=MemoryTier.M1,
                    memory_type=MemoryType.OBSERVATION,
                    robot_id="go2_test",
                    waypoint=None,
                ),
                auto_link=False,
            )

        return mgr

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.return_value = (
            "Summary: This location has chairs and a table.\n"
            "Keywords: kitchen, furniture, chairs"
        )
        return llm

    def test_groups_by_waypoint(self, manager, mock_llm, processed_index):
        """min_group_size=3 should produce summaries for both waypoint groups."""
        plugin = SpatialConsolidation(min_group_size=3)
        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(new_ids) >= 2

    def test_preserves_waypoint(self, manager, mock_llm, processed_index):
        """Created M2 items should preserve the waypoint from the source group."""
        plugin = SpatialConsolidation(min_group_size=3)
        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)

        waypoints_found = set()
        for item_id in new_ids:
            item = manager.get(item_id)
            assert item is not None
            assert item.waypoint is not None
            waypoints_found.add(item.waypoint)

        assert 5 in waypoints_found or 10 in waypoints_found

    def test_ignores_none_waypoint(self, tmp_path, mock_llm):
        """Items with waypoint=None should not form groups."""
        mgr = MemoryManager(persist_dir=str(tmp_path / "chroma_none"), robot_id="go2_test")
        pi = ProcessedIndex(str(tmp_path / "processed_none.json"))
        for i in range(5):
            mgr.add(
                MemoryItem(
                    document=f"Observation without location {i}",
                    tier=MemoryTier.M1,
                    memory_type=MemoryType.OBSERVATION,
                    robot_id="go2_test",
                    waypoint=None,
                ),
                auto_link=False,
            )

        plugin = SpatialConsolidation(min_group_size=3)
        new_ids = plugin.run(mgr, llm=mock_llm, processed_index=pi)
        assert new_ids == []

    def test_skips_small_groups(self, manager, mock_llm, processed_index):
        """min_group_size=10 means no group qualifies."""
        plugin = SpatialConsolidation(min_group_size=10)
        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert new_ids == []

    def test_no_llm_returns_empty(self, manager, processed_index):
        """When llm=None, run() should return an empty list immediately."""
        plugin = SpatialConsolidation(min_group_size=3)
        new_ids = plugin.run(manager, llm=None, processed_index=processed_index)
        assert new_ids == []

    def test_does_not_reprocess(self, manager, mock_llm, processed_index):
        """Second run should return [] since items are marked processed."""
        plugin = SpatialConsolidation(min_group_size=3)
        first_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(first_ids) >= 2

        second_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert second_ids == []

    def test_keywords_from_llm(self, manager, mock_llm, processed_index):
        """Created M2 items should have keywords parsed from the LLM response."""
        plugin = SpatialConsolidation(min_group_size=3)
        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(new_ids) > 0

        item = manager.get(new_ids[0])
        assert item is not None
        assert "kitchen" in item.keywords
        assert "furniture" in item.keywords
        assert "chairs" in item.keywords
