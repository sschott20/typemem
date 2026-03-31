"""Tests for the TaskPatternExtractor consolidation plugin."""

import pytest
from unittest.mock import MagicMock

from typemem.plugins.consolidation.task_pattern_extractor import TaskPatternExtractor
from typemem.memory_manager import MemoryManager
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.processed_index import ProcessedIndex


class TestTaskPatternExtractor:

    @pytest.fixture
    def processed_index(self, tmp_path):
        return ProcessedIndex(str(tmp_path / "processed.json"))

    @pytest.fixture
    def manager(self, tmp_path):
        mgr = MemoryManager(persist_dir=str(tmp_path / "chroma"), robot_id="go2_test")
        # 5 matching task items (ACTION type, "Task completed/stopped:" prefix)
        mgr.add(MemoryItem(
            document="Task completed: patrol the hallway",
            tier=MemoryTier.M1,
            memory_type=MemoryType.ACTION,
            robot_id="go2_test",
        ), auto_link=False)
        mgr.add(MemoryItem(
            document="Task completed: check the kitchen",
            tier=MemoryTier.M1,
            memory_type=MemoryType.ACTION,
            robot_id="go2_test",
        ), auto_link=False)
        mgr.add(MemoryItem(
            document="Task completed: walk to room A",
            tier=MemoryTier.M1,
            memory_type=MemoryType.ACTION,
            robot_id="go2_test",
        ), auto_link=False)
        mgr.add(MemoryItem(
            document="Task stopped: find the ball",
            tier=MemoryTier.M1,
            memory_type=MemoryType.ACTION,
            robot_id="go2_test",
        ), auto_link=False)
        mgr.add(MemoryItem(
            document="Task completed: patrol corridor B",
            tier=MemoryTier.M1,
            memory_type=MemoryType.ACTION,
            robot_id="go2_test",
        ), auto_link=False)
        # 2 non-matching items (ACTION type but wrong prefix)
        mgr.add(MemoryItem(
            document="Action completed: move forward",
            tier=MemoryTier.M1,
            memory_type=MemoryType.ACTION,
            robot_id="go2_test",
        ), auto_link=False)
        mgr.add(MemoryItem(
            document="Action completed: turn left",
            tier=MemoryTier.M1,
            memory_type=MemoryType.ACTION,
            robot_id="go2_test",
        ), auto_link=False)
        return mgr

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.return_value = (
            "Summary: Tasks related to patrol and navigation complete successfully. "
            "The robot handles hallway and corridor patrols reliably.\n"
            "Keywords: patrol, navigation, completion, success"
        )
        return llm

    def test_groups_task_patterns(self, manager, mock_llm, processed_index):
        plugin = TaskPatternExtractor(min_group_size=3)
        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(new_ids) >= 1
        # Verify M2 SUMMARY was created
        m2_items = manager.get_by_tier(MemoryTier.M2)
        assert len(m2_items) >= 1
        assert m2_items[0].memory_type == MemoryType.SUMMARY

    def test_filters_non_task_items(self, tmp_path):
        """Only 2 matching task items + 5 non-matching -> below min_group_size=3."""
        mgr = MemoryManager(persist_dir=str(tmp_path / "chroma"), robot_id="go2_test")
        pi = ProcessedIndex(str(tmp_path / "processed_filter.json"))
        # 2 matching task items
        mgr.add(MemoryItem(
            document="Task completed: patrol the hallway",
            tier=MemoryTier.M1,
            memory_type=MemoryType.ACTION,
            robot_id="go2_test",
        ), auto_link=False)
        mgr.add(MemoryItem(
            document="Task stopped: find the ball",
            tier=MemoryTier.M1,
            memory_type=MemoryType.ACTION,
            robot_id="go2_test",
        ), auto_link=False)
        # 5 non-matching action items
        for i in range(5):
            mgr.add(MemoryItem(
                document=f"Action completed: step {i}",
                tier=MemoryTier.M1,
                memory_type=MemoryType.ACTION,
                robot_id="go2_test",
            ), auto_link=False)

        mock_llm = MagicMock()
        plugin = TaskPatternExtractor(min_group_size=3)
        new_ids = plugin.run(mgr, llm=mock_llm, processed_index=pi)
        assert new_ids == []

    def test_skips_small_groups(self, manager, mock_llm, processed_index):
        plugin = TaskPatternExtractor(min_group_size=10)
        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert new_ids == []

    def test_no_llm_returns_empty(self, manager, processed_index):
        plugin = TaskPatternExtractor(min_group_size=3)
        new_ids = plugin.run(manager, llm=None, processed_index=processed_index)
        assert new_ids == []

    def test_does_not_reprocess(self, manager, mock_llm, processed_index):
        plugin = TaskPatternExtractor(min_group_size=3)
        first_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(first_ids) >= 1
        second_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert second_ids == []

    def test_keywords_from_llm(self, manager, mock_llm, processed_index):
        plugin = TaskPatternExtractor(min_group_size=3)
        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(new_ids) >= 1
        m2_items = manager.get_by_tier(MemoryTier.M2)
        keywords = m2_items[0].keywords
        assert "patrol" in keywords
        assert "navigation" in keywords
