"""Test that multiple plugins can independently process the same source items."""

import time
import pytest
from unittest.mock import MagicMock

from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex
from typemem.plugins.consolidation.m1_to_m2 import M1ToM2Strategy
from typemem.plugins.consolidation.caption_spatial import CaptionSpatialConsolidator
from typemem.plugins.consolidation.caption_activity import CaptionActivityConsolidator


class TestCrossPluginContention:
    @pytest.fixture
    def manager(self, tmp_path):
        return MemoryManager(persist_dir=str(tmp_path / "chroma"), robot_id="test")

    @pytest.fixture
    def processed_index(self, tmp_path):
        return ProcessedIndex(str(tmp_path / "processed.json"))

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.return_value = (
            "Summary: Objects observed in a room.\n"
            "Keywords: room, objects"
        )
        return llm

    def _seed_vlm_captions(self, manager, n=5, waypoint=3):
        ids = []
        for i in range(n):
            item = MemoryItem(
                document=f"Camera shows object {i} near the table in room.",
                tier=MemoryTier.M1,
                memory_type=MemoryType.OBSERVATION,
                robot_id="test",
                waypoint=waypoint,
                keywords="vlm_caption",
                timestamp=time.time() + i,
            )
            manager._collection.add(
                ids=[item.id], documents=[item.document],
                metadatas=[item.to_metadata()],
            )
            ids.append(item.id)
        return ids

    def test_first_plugin_does_not_starve_others(
        self, manager, processed_index, mock_llm,
    ):
        """After M1ToM2 processes items, CaptionSpatial still sees them."""
        self._seed_vlm_captions(manager, n=5, waypoint=3)

        m1m2 = M1ToM2Strategy(min_group_size=3)
        m1m2.run(manager, llm=mock_llm, processed_index=processed_index)

        spatial = CaptionSpatialConsolidator(min_group_size=3)
        spatial_ids = spatial.run(
            manager, llm=mock_llm, processed_index=processed_index,
        )
        assert len(spatial_ids) >= 1, "CaptionSpatial was starved by M1ToM2"

    def test_all_three_plugins_process_same_items(
        self, manager, processed_index, mock_llm,
    ):
        """M1ToM2, CaptionSpatial, CaptionActivity all process the same M1 items."""
        self._seed_vlm_captions(manager, n=5, waypoint=2)

        m1m2 = M1ToM2Strategy(min_group_size=3)
        spatial = CaptionSpatialConsolidator(min_group_size=3)
        activity = CaptionActivityConsolidator(min_items=3)

        ids1 = m1m2.run(manager, llm=mock_llm, processed_index=processed_index)
        ids2 = spatial.run(manager, llm=mock_llm, processed_index=processed_index)
        ids3 = activity.run(manager, llm=mock_llm, processed_index=processed_index)

        assert len(ids1) >= 1
        assert len(ids2) >= 1
        assert len(ids3) >= 1

    def test_same_plugin_does_not_reprocess(
        self, manager, processed_index, mock_llm,
    ):
        """A single plugin should not process the same items twice."""
        self._seed_vlm_captions(manager, n=5, waypoint=1)

        spatial = CaptionSpatialConsolidator(min_group_size=3)
        first = spatial.run(manager, llm=mock_llm, processed_index=processed_index)
        second = spatial.run(manager, llm=mock_llm, processed_index=processed_index)

        assert len(first) >= 1
        assert second == []
