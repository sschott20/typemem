"""Tests for the TemporalPatternExtractor consolidation plugin."""

import datetime
import pytest
from unittest.mock import MagicMock

from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex
from typemem.plugins.consolidation.temporal_pattern_extractor import TemporalPatternExtractor


class TestTemporalPatternExtractor:

    @pytest.fixture
    def processed_index(self, tmp_path):
        return ProcessedIndex(str(tmp_path / "processed.json"))

    @pytest.fixture
    def manager(self, tmp_path):
        mgr = MemoryManager(persist_dir=str(tmp_path / "chroma"), robot_id="go2_test")

        # 3 morning items (8am, 9am, 10am)
        for hour in (8, 9, 10):
            ts = datetime.datetime(2024, 1, 1, hour, 0).timestamp()
            item = MemoryItem(
                document=f"Morning observation at {hour}am",
                tier=MemoryTier.M2,
                memory_type=MemoryType.SUMMARY,
                robot_id="go2_test",
                timestamp=ts,
            )
            mgr.add(item, auto_link=False)

        # 2 evening items (7pm, 8pm)
        for hour in (19, 20):
            ts = datetime.datetime(2024, 1, 1, hour, 0).timestamp()
            item = MemoryItem(
                document=f"Evening observation at {hour - 12}pm",
                tier=MemoryTier.M2,
                memory_type=MemoryType.SUMMARY,
                robot_id="go2_test",
                timestamp=ts,
            )
            mgr.add(item, auto_link=False)

        return mgr

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.return_value = (
            "Lesson: During morning hours, the kitchen area is typically unoccupied.\n"
            "Keywords: morning, kitchen, empty, temporal"
        )
        return llm

    def test_bins_by_time_of_day(self, manager, mock_llm, processed_index):
        """Morning bin (3 items) and evening bin (2 items) both qualify."""
        plugin = TemporalPatternExtractor(min_summaries=3, min_bin_size=2)
        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(new_ids) >= 1

    def test_too_few_summaries(self, manager, mock_llm, processed_index):
        """If min_summaries is higher than available items, returns empty."""
        plugin = TemporalPatternExtractor(min_summaries=10, min_bin_size=2)
        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert new_ids == []

    def test_small_bins_skipped(self, manager, mock_llm, processed_index):
        """If min_bin_size is too large, no bins qualify."""
        plugin = TemporalPatternExtractor(min_summaries=3, min_bin_size=10)
        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert new_ids == []

    def test_no_llm_returns_empty(self, manager, processed_index):
        """Without an LLM, the plugin returns empty."""
        plugin = TemporalPatternExtractor(min_summaries=3, min_bin_size=2)
        new_ids = plugin.run(manager, llm=None, processed_index=processed_index)
        assert new_ids == []

    def test_does_not_reprocess(self, manager, mock_llm, processed_index):
        """Second run returns empty because items are already processed."""
        plugin = TemporalPatternExtractor(min_summaries=3, min_bin_size=2)
        first_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(first_ids) >= 1
        second_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert second_ids == []

    def test_keywords_from_llm(self, manager, mock_llm, processed_index):
        """Created M3 item has keywords parsed from LLM response."""
        plugin = TemporalPatternExtractor(min_summaries=3, min_bin_size=2)
        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed_index)
        assert len(new_ids) >= 1
        item = manager.get(new_ids[0])
        assert item is not None
        keywords = item.keywords.split(",")
        assert "morning" in keywords
        assert "kitchen" in keywords
        assert "temporal" in keywords
