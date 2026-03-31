import pytest
from unittest.mock import MagicMock
from typemem.plugins.consolidation.m1_to_m2 import M1ToM2Strategy
from typemem.memory_manager import MemoryManager
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.processed_index import ProcessedIndex


class TestM1ToM2Strategy:
    @pytest.fixture
    def processed_index(self, tmp_path):
        return ProcessedIndex(str(tmp_path / "processed.json"))

    @pytest.fixture
    def manager(self, tmp_path):
        mgr = MemoryManager(persist_dir=str(tmp_path / "chroma"), robot_id="go2_test")
        # Add several M1 observations about the same topic
        # Bypass dedup gate by adding directly to collection
        for i in range(5):
            item = MemoryItem(
                document=f"I saw a chair near the kitchen at waypoint 5, observation {i}",
                tier=MemoryTier.M1,
                memory_type=MemoryType.OBSERVATION,
                robot_id="go2_test",
                waypoint=5,
                keywords="chair,kitchen",
            )
            mgr._collection.add(
                ids=[item.id], documents=[item.document], metadatas=[item.to_metadata()],
            )
        # Add an unrelated observation
        item = MemoryItem(
            document="A ball was found on the soccer field",
            tier=MemoryTier.M1,
            memory_type=MemoryType.OBSERVATION,
            robot_id="go2_test",
            waypoint=10,
            keywords="ball,field",
        )
        mgr._collection.add(
            ids=[item.id], documents=[item.document], metadatas=[item.to_metadata()],
        )
        return mgr

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.return_value = "Summary: A chair has been consistently observed near the kitchen at waypoint 5.\nKeywords: chair, kitchen, waypoint 5"
        return llm

    def test_groups_similar_observations(self, manager, mock_llm, processed_index):
        strategy = M1ToM2Strategy(min_group_size=3)
        new_ids = strategy.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(new_ids) >= 1

        # Verify M2 item was created
        m2_item = manager.get(new_ids[0])
        assert m2_item.tier == MemoryTier.M2
        assert m2_item.memory_type == MemoryType.SUMMARY

    def test_llm_receives_grouped_observations(self, manager, mock_llm, processed_index):
        strategy = M1ToM2Strategy(min_group_size=3)
        strategy.run(manager, llm=mock_llm, processed_index=processed_index)
        # LLM should have been called with the grouped observations
        assert mock_llm.called

    def test_skips_small_groups(self, manager, mock_llm, processed_index):
        strategy = M1ToM2Strategy(min_group_size=10)  # require 10 items per group
        new_ids = strategy.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(new_ids) == 0  # no group has 10 items

    def test_no_llm_returns_empty(self, manager, processed_index):
        strategy = M1ToM2Strategy(min_group_size=3)
        new_ids = strategy.run(manager, llm=None, processed_index=processed_index)
        assert len(new_ids) == 0

    def test_does_not_reprocess(self, manager, mock_llm, processed_index):
        strategy = M1ToM2Strategy(min_group_size=3)
        first_run = strategy.run(manager, llm=mock_llm, processed_index=processed_index)
        second_run = strategy.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(first_run) >= 1
        assert len(second_run) == 0  # already processed

    def test_keywords_aggregated(self, manager, mock_llm, processed_index):
        strategy = M1ToM2Strategy(min_group_size=3)
        new_ids = strategy.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(new_ids) >= 1
        m2_item = manager.get(new_ids[0])
        # Should have aggregated keywords from the group
        assert "chair" in m2_item.keywords or "kitchen" in m2_item.keywords
