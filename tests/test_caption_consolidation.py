"""Tests for VLM caption consolidation plugins (spatial + activity)."""

import time
import pytest
from typing import List
from unittest.mock import MagicMock

from typemem.plugins.base import ConsolidationPlugin
from typemem.plugins.consolidation.caption_spatial import CaptionSpatialConsolidator
from typemem.plugins.consolidation.caption_activity import CaptionActivityConsolidator
from typemem.consolidation import ConsolidationEngine
from typemem.memory_manager import MemoryManager
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.processed_index import ProcessedIndex


def _seed_vlm_captions(manager, n=5, waypoint=3):
    """Add n VLM caption M1 items to the manager.

    Bypasses the dedup gate by adding directly to the collection,
    since these test items are intentionally near-similar.
    """
    ids = []
    for i in range(n):
        item = MemoryItem(
            document=f"A room with object {i} near the table.",
            tier=MemoryTier.M1,
            memory_type=MemoryType.OBSERVATION,
            robot_id="test",
            waypoint=waypoint,
            keywords="vlm_caption",
            timestamp=time.time() + i,  # sequential timestamps
        )
        manager._collection.add(
            ids=[item.id], documents=[item.document], metadatas=[item.to_metadata()],
        )
        ids.append(item.id)
    return ids


def _seed_non_vlm_items(manager, n=3):
    """Add non-VLM M1 items (scene graph style) -- no vlm_caption keyword.

    Bypasses the dedup gate by adding directly to the collection.
    """
    for i in range(n):
        item = MemoryItem(
            document=f"Object {i} detected at position [1, 2, 3].",
            tier=MemoryTier.M1,
            memory_type=MemoryType.OBSERVATION,
            robot_id="test",
            waypoint=1,
            keywords="scene_graph",
        )
        manager._collection.add(
            ids=[item.id], documents=[item.document], metadatas=[item.to_metadata()],
        )


@pytest.fixture
def processed_index(tmp_path):
    return ProcessedIndex(str(tmp_path / "processed.json"))


@pytest.fixture
def manager(tmp_path):
    return MemoryManager(persist_dir=str(tmp_path / "chroma"), robot_id="test")


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.return_value = (
        "Summary: A table with chairs is in the center of the room near the window.\n"
        "Keywords: table, chairs, window, room"
    )
    return llm


# ===========================================================================
# CaptionSpatialConsolidator
# ===========================================================================

class TestCaptionSpatialConsolidator:
    def test_is_consolidation_plugin(self):
        assert isinstance(CaptionSpatialConsolidator(), ConsolidationPlugin)

    def test_properties(self):
        c = CaptionSpatialConsolidator()
        assert c.name == "CaptionSpatial"
        assert c.source_tier == MemoryTier.M1
        assert c.target_tier == MemoryTier.M2
        assert c.interval_seconds == 60.0

    def test_returns_empty_without_llm(self, manager, processed_index):
        _seed_vlm_captions(manager, n=5)
        c = CaptionSpatialConsolidator()
        assert c.run(manager, llm=None, processed_index=processed_index) == []

    def test_returns_empty_below_threshold(self, manager, mock_llm, processed_index):
        _seed_vlm_captions(manager, n=2)
        c = CaptionSpatialConsolidator(min_group_size=3)
        assert c.run(manager, llm=mock_llm, processed_index=processed_index) == []

    def test_creates_m2_spatial_summary(self, manager, mock_llm, processed_index):
        _seed_vlm_captions(manager, n=5, waypoint=3)
        c = CaptionSpatialConsolidator(min_group_size=3)

        ids = c.run(manager, llm=mock_llm, processed_index=processed_index)

        assert len(ids) == 1
        item = manager.get(ids[0])
        assert item.tier == MemoryTier.M2
        assert item.memory_type == MemoryType.SUMMARY
        assert "spatial" in item.keywords
        assert "vlm_caption" in item.keywords
        assert item.waypoint == 3

    def test_ignores_non_vlm_items(self, manager, mock_llm, processed_index):
        """Only processes items with vlm_caption keyword."""
        _seed_non_vlm_items(manager, n=5)
        c = CaptionSpatialConsolidator(min_group_size=3)

        ids = c.run(manager, llm=mock_llm, processed_index=processed_index)
        assert ids == []

    def test_marks_source_processed(self, manager, mock_llm, processed_index):
        _seed_vlm_captions(manager, n=4, waypoint=1)
        c = CaptionSpatialConsolidator(min_group_size=3)

        c.run(manager, llm=mock_llm, processed_index=processed_index)

        all_m1 = manager.get_by_tier(MemoryTier.M1)
        for item in all_m1:
            if "vlm_caption" in item.keywords:
                assert processed_index.is_processed("CaptionSpatial", item.id)

    def test_does_not_reprocess(self, manager, mock_llm, processed_index):
        """Second run produces nothing (items already processed)."""
        _seed_vlm_captions(manager, n=5, waypoint=2)
        c = CaptionSpatialConsolidator(min_group_size=3)

        c.run(manager, llm=mock_llm, processed_index=processed_index)
        ids2 = c.run(manager, llm=mock_llm, processed_index=processed_index)

        assert ids2 == []

    def test_groups_by_waypoint(self, manager, mock_llm, processed_index):
        """Items at different waypoints form separate groups."""
        _seed_vlm_captions(manager, n=3, waypoint=1)
        _seed_vlm_captions(manager, n=3, waypoint=2)
        c = CaptionSpatialConsolidator(min_group_size=3)

        ids = c.run(manager, llm=mock_llm, processed_index=processed_index)

        assert len(ids) == 2

    def test_registers_in_engine(self, manager):
        engine = ConsolidationEngine(manager)
        engine.register_strategy(CaptionSpatialConsolidator())
        assert "CaptionSpatial" in engine.list_strategies()


# ===========================================================================
# CaptionActivityConsolidator
# ===========================================================================

class TestCaptionActivityConsolidator:
    def test_is_consolidation_plugin(self):
        assert isinstance(CaptionActivityConsolidator(), ConsolidationPlugin)

    def test_properties(self):
        c = CaptionActivityConsolidator()
        assert c.name == "CaptionActivity"
        assert c.source_tier == MemoryTier.M1
        assert c.target_tier == MemoryTier.M2
        assert c.interval_seconds == 30.0

    def test_returns_empty_without_llm(self, manager, processed_index):
        _seed_vlm_captions(manager, n=5)
        c = CaptionActivityConsolidator()
        assert c.run(manager, llm=None, processed_index=processed_index) == []

    def test_returns_empty_below_threshold(self, manager, mock_llm, processed_index):
        _seed_vlm_captions(manager, n=2)
        c = CaptionActivityConsolidator(min_items=3)
        assert c.run(manager, llm=mock_llm, processed_index=processed_index) == []

    def test_creates_m2_activity_summary(self, manager, mock_llm, processed_index):
        _seed_vlm_captions(manager, n=5, waypoint=2)
        c = CaptionActivityConsolidator(min_items=3)

        ids = c.run(manager, llm=mock_llm, processed_index=processed_index)

        assert len(ids) == 1
        item = manager.get(ids[0])
        assert item.tier == MemoryTier.M2
        assert item.memory_type == MemoryType.SUMMARY
        assert "activity" in item.keywords
        assert "vlm_caption" in item.keywords
        assert item.waypoint == 2

    def test_ignores_non_vlm_items(self, manager, mock_llm, processed_index):
        _seed_non_vlm_items(manager, n=5)
        c = CaptionActivityConsolidator(min_items=3)

        ids = c.run(manager, llm=mock_llm, processed_index=processed_index)
        assert ids == []

    def test_marks_source_processed(self, manager, mock_llm, processed_index):
        _seed_vlm_captions(manager, n=4)
        c = CaptionActivityConsolidator(min_items=3)

        c.run(manager, llm=mock_llm, processed_index=processed_index)

        all_m1 = manager.get_by_tier(MemoryTier.M1)
        for item in all_m1:
            if "vlm_caption" in item.keywords:
                assert processed_index.is_processed("CaptionActivity", item.id)

    def test_does_not_reprocess(self, manager, mock_llm, processed_index):
        _seed_vlm_captions(manager, n=5)
        c = CaptionActivityConsolidator(min_items=3)

        c.run(manager, llm=mock_llm, processed_index=processed_index)
        ids2 = c.run(manager, llm=mock_llm, processed_index=processed_index)

        assert ids2 == []

    def test_temporal_ordering_in_prompt(self, manager, mock_llm, processed_index):
        """Verify LLM receives observations in temporal order with indices."""
        _seed_vlm_captions(manager, n=3)
        c = CaptionActivityConsolidator(min_items=3)

        c.run(manager, llm=mock_llm, processed_index=processed_index)

        prompt = mock_llm.call_args[0][0]
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt

    def test_registers_in_engine(self, manager):
        engine = ConsolidationEngine(manager)
        engine.register_strategy(CaptionActivityConsolidator())
        assert "CaptionActivity" in engine.list_strategies()
