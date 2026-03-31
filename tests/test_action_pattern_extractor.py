import pytest
from unittest.mock import MagicMock
from typemem.plugins.consolidation.action_pattern_extractor import ActionPatternExtractor
from typemem.memory_manager import MemoryManager
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.processed_index import ProcessedIndex


class TestActionPatternExtractor:
    @pytest.fixture
    def processed_index(self, tmp_path):
        return ProcessedIndex(str(tmp_path / "processed.json"))

    @pytest.fixture
    def manager(self, tmp_path):
        mgr = MemoryManager(persist_dir=str(tmp_path / "chroma"), robot_id="go2_test")
        # Add 5 ACTION items matching the real observer format
        # Bypass dedup gate by adding directly to collection
        for i in range(5):
            item = MemoryItem(
                document=f"Action: walk(forward) | Result: SUCCESS | Duration: {i}.0s | Task: patrol hallway #{i}",
                tier=MemoryTier.M1,
                memory_type=MemoryType.ACTION,
                robot_id="go2_test",
                keywords="walk(forward),success",
            )
            mgr._collection.add(
                ids=[item.id], documents=[item.document], metadatas=[item.to_metadata()],
            )
        # Add a non-matching OBSERVATION item
        item = MemoryItem(
            document="I saw a chair near the kitchen",
            tier=MemoryTier.M1,
            memory_type=MemoryType.OBSERVATION,
            robot_id="go2_test",
            keywords="chair,kitchen",
        )
        mgr._collection.add(
            ids=[item.id], documents=[item.document], metadatas=[item.to_metadata()],
        )
        # Add an ACTION item that does not match the pattern
        item = MemoryItem(
            document="Robot performed an unknown maneuver",
            tier=MemoryTier.M1,
            memory_type=MemoryType.ACTION,
            robot_id="go2_test",
            keywords="unknown,action",
        )
        mgr._collection.add(
            ids=[item.id], documents=[item.document], metadatas=[item.to_metadata()],
        )
        return mgr

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.return_value = (
            "Summary: Walk actions succeed 80% of the time during PATROL state.\n"
            "Keywords: walk, movement, success rate"
        )
        return llm

    def test_groups_action_patterns(self, manager, mock_llm, processed_index):
        extractor = ActionPatternExtractor(min_group_size=3)
        new_ids = extractor.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(new_ids) >= 1

        # Verify M2 SUMMARY was created
        m2_item = manager.get(new_ids[0])
        assert m2_item.tier == MemoryTier.M2
        assert m2_item.memory_type == MemoryType.SUMMARY

    def test_filters_non_action_items(self, tmp_path, mock_llm):
        mgr = MemoryManager(persist_dir=str(tmp_path / "chroma_filter"), robot_id="go2_test")
        pi = ProcessedIndex(str(tmp_path / "processed_filter.json"))
        # Only 2 ACTION items (below threshold)
        # Bypass dedup gate by adding directly to collection
        for i in range(2):
            item = MemoryItem(
                document=f"Action: sit_down | Result: SUCCESS | Duration: 1.0s | Task: rest #{i}",
                tier=MemoryTier.M1,
                memory_type=MemoryType.ACTION,
                robot_id="go2_test",
                keywords="sit_down,success",
            )
            mgr._collection.add(
                ids=[item.id], documents=[item.document], metadatas=[item.to_metadata()],
            )
        # 5 OBSERVATION items (wrong type)
        for i in range(5):
            item = MemoryItem(
                document=f"Observation: temperature is {20 + i} degrees",
                tier=MemoryTier.M1,
                memory_type=MemoryType.OBSERVATION,
                robot_id="go2_test",
                keywords="temperature,observation",
            )
            mgr._collection.add(
                ids=[item.id], documents=[item.document], metadatas=[item.to_metadata()],
            )
        extractor = ActionPatternExtractor(min_group_size=3)
        new_ids = extractor.run(mgr, llm=mock_llm, processed_index=pi)
        assert new_ids == []

    def test_skips_small_groups(self, manager, mock_llm, processed_index):
        extractor = ActionPatternExtractor(min_group_size=10)
        new_ids = extractor.run(manager, llm=mock_llm, processed_index=processed_index)
        assert new_ids == []

    def test_no_llm_returns_empty(self, manager, processed_index):
        extractor = ActionPatternExtractor(min_group_size=3)
        new_ids = extractor.run(manager, llm=None, processed_index=processed_index)
        assert new_ids == []

    def test_does_not_reprocess(self, manager, mock_llm, processed_index):
        extractor = ActionPatternExtractor(min_group_size=3)
        first_run = extractor.run(manager, llm=mock_llm, processed_index=processed_index)
        second_run = extractor.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(first_run) >= 1
        assert len(second_run) == 0

    def test_keywords_from_llm(self, manager, mock_llm, processed_index):
        extractor = ActionPatternExtractor(min_group_size=3)
        new_ids = extractor.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(new_ids) >= 1
        m2_item = manager.get(new_ids[0])
        assert "walk" in m2_item.keywords
        assert "movement" in m2_item.keywords
        assert "success rate" in m2_item.keywords
