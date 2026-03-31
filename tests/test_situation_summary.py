import pytest
from unittest.mock import MagicMock
from typemem.memory_manager import MemoryManager
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.processed_index import ProcessedIndex
from typemem.plugins.consolidation.situation_summary import SituationSummaryPlugin


@pytest.fixture
def manager(tmp_path):
    return MemoryManager(persist_dir=str(tmp_path / "chroma"), robot_id="test")

@pytest.fixture
def processed(tmp_path):
    return ProcessedIndex(str(tmp_path / "processed.json"))


class TestSituationSummaryPlugin:
    def test_creates_m2_summary_from_m1_items(self, manager, processed):
        for i in range(3):
            manager.add(MemoryItem(document=f"saw object {i}", tier=MemoryTier.M1, memory_type=MemoryType.OBSERVATION, robot_id="test"))

        mock_llm = MagicMock(return_value="A robot observing objects in a room.")
        plugin = SituationSummaryPlugin()
        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed)

        assert len(new_ids) == 1
        items = manager.get_by_source("situation_summary", tier=MemoryTier.M2)
        assert len(items) == 1
        assert "observing" in items[0].document.lower()

    def test_replaces_previous_summary(self, manager, processed):
        manager.add(MemoryItem(document="old summary", tier=MemoryTier.M2, memory_type=MemoryType.SUMMARY, robot_id="test", source="situation_summary"))
        manager.add(MemoryItem(document="new observation", tier=MemoryTier.M1, memory_type=MemoryType.OBSERVATION, robot_id="test"))

        mock_llm = MagicMock(return_value="Updated summary.")
        plugin = SituationSummaryPlugin()
        plugin.run(manager, llm=mock_llm, processed_index=processed)

        items = manager.get_by_source("situation_summary", tier=MemoryTier.M2)
        assert len(items) == 1
        assert items[0].document == "Updated summary."

    def test_llm_receives_previous_summary(self, manager, processed):
        manager.add(MemoryItem(document="previous state", tier=MemoryTier.M2, memory_type=MemoryType.SUMMARY, robot_id="test", source="situation_summary"))
        manager.add(MemoryItem(document="new obs", tier=MemoryTier.M1, memory_type=MemoryType.OBSERVATION, robot_id="test"))

        mock_llm = MagicMock(return_value="Updated.")
        plugin = SituationSummaryPlugin()
        plugin.run(manager, llm=mock_llm, processed_index=processed)

        prompt = mock_llm.call_args[0][0]
        assert "previous state" in prompt

    def test_skips_when_no_new_items(self, manager, processed):
        mock_llm = MagicMock()
        plugin = SituationSummaryPlugin()
        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed)
        assert new_ids == []
        mock_llm.assert_not_called()

    def test_skips_without_llm(self, manager, processed):
        manager.add(MemoryItem(document="obs", tier=MemoryTier.M1, memory_type=MemoryType.OBSERVATION, robot_id="test"))
        plugin = SituationSummaryPlugin()
        new_ids = plugin.run(manager, llm=None, processed_index=processed)
        assert new_ids == []
