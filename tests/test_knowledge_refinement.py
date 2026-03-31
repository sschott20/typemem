"""Tests for KnowledgeRefinement consolidation plugin (M3 -> M3 dedup/merge)."""

import pytest
from unittest.mock import MagicMock

from typemem.memory_manager import MemoryManager
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.processed_index import ProcessedIndex
from typemem.plugins.consolidation.knowledge_refinement import KnowledgeRefinement


class TestKnowledgeRefinement:
    @pytest.fixture
    def processed_index(self, tmp_path):
        return ProcessedIndex(str(tmp_path / "processed.json"))

    @pytest.fixture
    def manager(self, tmp_path):
        mgr = MemoryManager(persist_dir=str(tmp_path / "chroma"), robot_id="go2_test")
        # Three nearly identical kitchen observations
        mgr.add(MemoryItem(
            document="The kitchen is usually empty in the morning",
            tier=MemoryTier.M3,
            memory_type=MemoryType.LESSON,
            robot_id="go2_test",
            keywords="kitchen,morning,empty",
        ), auto_link=False)
        mgr.add(MemoryItem(
            document="The kitchen area is empty during morning hours",
            tier=MemoryTier.M3,
            memory_type=MemoryType.LESSON,
            robot_id="go2_test",
            keywords="kitchen,morning,empty",
        ), auto_link=False)
        mgr.add(MemoryItem(
            document="Morning kitchen observations show no people present",
            tier=MemoryTier.M3,
            memory_type=MemoryType.LESSON,
            robot_id="go2_test",
            keywords="kitchen,morning,people",
        ), auto_link=False)
        # One distinct item
        mgr.add(MemoryItem(
            document="The robot charges fastest at the docking station near room B",
            tier=MemoryTier.M3,
            memory_type=MemoryType.LESSON,
            robot_id="go2_test",
            keywords="charging,docking,room_b",
        ), auto_link=False)
        return mgr

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.return_value = (
            "Lesson: The kitchen is consistently empty during morning hours.\n"
            "Keywords: kitchen, morning, empty, schedule"
        )
        return llm

    @pytest.fixture
    def mock_llm_keep_all(self):
        llm = MagicMock()
        llm.return_value = "KEEP_ALL"
        return llm

    def test_merges_similar_knowledge(self, manager, mock_llm, processed_index):
        plugin = KnowledgeRefinement()
        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(new_ids) >= 1
        # Verify new M3 LESSON was created
        for nid in new_ids:
            item = manager.get(nid)
            assert item is not None
            assert item.tier == MemoryTier.M3
            assert item.memory_type == MemoryType.LESSON

    def test_keep_all_skips_group(self, manager, mock_llm_keep_all, processed_index):
        plugin = KnowledgeRefinement()
        new_ids = plugin.run(manager, llm=mock_llm_keep_all, processed_index=processed_index)
        assert new_ids == []
        # Items should NOT be marked processed
        all_m3 = manager.get_by_tier(MemoryTier.M3)
        for item in all_m3:
            assert not processed_index.is_processed("KnowledgeRefinement", item.id)

    def test_no_llm_returns_empty(self, manager, processed_index):
        plugin = KnowledgeRefinement()
        new_ids = plugin.run(manager, llm=None, processed_index=processed_index)
        assert new_ids == []

    def test_too_few_items(self, tmp_path):
        mgr = MemoryManager(persist_dir=str(tmp_path / "chroma_few"), robot_id="go2_test")
        pi = ProcessedIndex(str(tmp_path / "processed_few.json"))
        mgr.add(MemoryItem(
            document="Only one memory here",
            tier=MemoryTier.M3,
            memory_type=MemoryType.LESSON,
            robot_id="go2_test",
        ), auto_link=False)
        plugin = KnowledgeRefinement()
        new_ids = plugin.run(mgr, llm=MagicMock(), processed_index=pi)
        assert new_ids == []

    def test_does_not_reprocess(self, manager, mock_llm, processed_index):
        plugin = KnowledgeRefinement()
        first_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(first_ids) >= 1
        # Second run should find no unprocessed groups to merge
        second_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert second_ids == []

    def test_keywords_from_llm(self, manager, mock_llm, processed_index):
        plugin = KnowledgeRefinement()
        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(new_ids) >= 1
        item = manager.get(new_ids[0])
        assert item is not None
        assert "kitchen" in item.keywords
        assert "morning" in item.keywords
