import pytest
from unittest.mock import MagicMock
from typemem.plugins.consolidation.m2_to_m3 import M2ToM3Strategy
from typemem.memory_manager import MemoryManager
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.processed_index import ProcessedIndex


class TestM2ToM3Strategy:
    @pytest.fixture
    def processed_index(self, tmp_path):
        return ProcessedIndex(str(tmp_path / "processed.json"))

    @pytest.fixture
    def manager(self, tmp_path):
        mgr = MemoryManager(persist_dir=str(tmp_path / "chroma"), robot_id="go2_test")
        # Add M2 summaries spanning time
        # Bypass dedup gate by adding directly to collection
        for i in range(4):
            item = MemoryItem(
                document=f"Chairs have been consistently near waypoint 5 during period {i}",
                tier=MemoryTier.M2,
                memory_type=MemoryType.SUMMARY,
                robot_id="go2_test",
                waypoint=5,
                keywords="chair,waypoint_5",
            )
            mgr._collection.add(
                ids=[item.id], documents=[item.document], metadatas=[item.to_metadata()],
            )
        return mgr

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.return_value = "Lesson: Chairs are a permanent fixture near waypoint 5.\nKeywords: chair, waypoint 5, furniture"
        return llm

    def test_extracts_lesson_from_summaries(self, manager, mock_llm, processed_index):
        strategy = M2ToM3Strategy(min_summaries=3)
        new_ids = strategy.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(new_ids) >= 1
        m3_item = manager.get(new_ids[0])
        assert m3_item.tier == MemoryTier.M3
        assert m3_item.memory_type == MemoryType.LESSON

    def test_skips_if_too_few_summaries(self, manager, mock_llm, processed_index):
        strategy = M2ToM3Strategy(min_summaries=10)
        new_ids = strategy.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(new_ids) == 0

    def test_no_llm_returns_empty(self, manager, processed_index):
        strategy = M2ToM3Strategy(min_summaries=3)
        new_ids = strategy.run(manager, llm=None, processed_index=processed_index)
        assert len(new_ids) == 0

    def test_does_not_reprocess(self, manager, mock_llm, processed_index):
        strategy = M2ToM3Strategy(min_summaries=3)
        first_run = strategy.run(manager, llm=mock_llm, processed_index=processed_index)
        second_run = strategy.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(first_run) >= 1
        assert len(second_run) == 0

    def test_keywords_aggregated(self, manager, mock_llm, processed_index):
        strategy = M2ToM3Strategy(min_summaries=3)
        new_ids = strategy.run(manager, llm=mock_llm, processed_index=processed_index)
        m3_item = manager.get(new_ids[0])
        assert "chair" in m3_item.keywords

    def test_bounded_batching(self, tmp_path, mock_llm):
        """When items exceed max_batch_size, they're processed in multiple batches."""
        mgr = MemoryManager(persist_dir=str(tmp_path / "chroma_batch"), robot_id="go2_test")
        pi = ProcessedIndex(str(tmp_path / "processed_batch.json"))
        # Add 5 items, set batch size to 2
        for i in range(5):
            item = MemoryItem(
                document=f"Summary about topic {i} with unique content {i * 100}",
                tier=MemoryTier.M2,
                memory_type=MemoryType.SUMMARY,
                robot_id="go2_test",
                keywords=f"topic_{i}",
            )
            mgr._collection.add(
                ids=[item.id], documents=[item.document], metadatas=[item.to_metadata()],
            )
        strategy = M2ToM3Strategy(min_summaries=2, max_batch_size=2)
        new_ids = strategy.run(mgr, llm=mock_llm, processed_index=pi)
        # 5 items / batch size 2 = 3 batches (2, 2, 1)
        assert len(new_ids) == 3
        # LLM should have been called 3 times (once per batch)
        assert mock_llm.call_count == 3
