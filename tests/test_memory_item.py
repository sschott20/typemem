import time
import pytest
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType


class TestMemoryTier:
    def test_tier_labels(self):
        assert str(MemoryTier.M0) == "M0"
        assert str(MemoryTier.M1) == "M1"
        assert str(MemoryTier.M2) == "M2"
        assert str(MemoryTier.M3) == "M3"

    def test_tier_retention(self):
        assert MemoryTier.M0.retention == 60
        assert MemoryTier.M1.retention == 600
        assert MemoryTier.M2.retention == 3600
        assert MemoryTier.M3.retention is None

    def test_tier_levels_ordered(self):
        assert MemoryTier.M0.level < MemoryTier.M1.level
        assert MemoryTier.M1.level < MemoryTier.M2.level
        assert MemoryTier.M2.level < MemoryTier.M3.level


class TestMemoryType:
    def test_type_values(self):
        assert MemoryType.OBSERVATION.value == "observation"
        assert MemoryType.SUMMARY.value == "summary"
        assert MemoryType.INSTRUCTION.value == "instruction"
        assert MemoryType.ACTION.value == "action"
        assert MemoryType.LESSON.value == "lesson"


class TestMemoryItem:
    def test_create_basic(self):
        item = MemoryItem(document="saw a chair", tier=MemoryTier.M1, memory_type=MemoryType.OBSERVATION, robot_id="robot1")
        assert item.document == "saw a chair"
        assert item.tier == MemoryTier.M1
        assert item.memory_type == MemoryType.OBSERVATION
        assert item.robot_id == "robot1"
        assert item.id is not None
        assert item.timestamp > 0

    def test_to_metadata(self):
        item = MemoryItem(document="test", tier=MemoryTier.M2, memory_type=MemoryType.SUMMARY, robot_id="r1", waypoint=3, task_id=42, keywords="chair,table", source="scene_graph")
        meta = item.to_metadata()
        assert meta["tier"] == "M2"
        assert meta["memory_type"] == "summary"
        assert meta["robot_id"] == "r1"
        assert meta["waypoint"] == 3
        assert meta["task_id"] == 42
        assert meta["keywords"] == "chair,table"
        assert meta["source"] == "scene_graph"

    def test_to_metadata_omits_none_optionals(self):
        item = MemoryItem(document="test", tier=MemoryTier.M1, memory_type=MemoryType.OBSERVATION, robot_id="r1")
        meta = item.to_metadata()
        assert "waypoint" not in meta
        assert "task_id" not in meta

    def test_from_chromadb_roundtrip(self):
        item = MemoryItem(document="saw a ball", tier=MemoryTier.M1, memory_type=MemoryType.OBSERVATION, robot_id="r1", waypoint=5, task_id=10, keywords="ball", source="vision")
        meta = item.to_metadata()
        restored = MemoryItem.from_chromadb(item.id, item.document, meta)
        assert restored.id == item.id
        assert restored.document == item.document
        assert restored.tier == item.tier
        assert restored.memory_type == item.memory_type
        assert restored.robot_id == item.robot_id
        assert restored.waypoint == item.waypoint
        assert restored.task_id == item.task_id
        assert restored.keywords == item.keywords
        assert restored.source == item.source

    def test_unique_ids(self):
        a = MemoryItem(document="a", tier=MemoryTier.M0, memory_type=MemoryType.OBSERVATION, robot_id="r")
        b = MemoryItem(document="b", tier=MemoryTier.M0, memory_type=MemoryType.OBSERVATION, robot_id="r")
        assert a.id != b.id

    def test_frame_ref_roundtrip(self):
        item = MemoryItem(
            document="Chair seen near kitchen",
            tier=MemoryTier.M1,
            memory_type=MemoryType.OBSERVATION,
            robot_id="test",
            frame_ref="f_abc123",
        )
        meta = item.to_metadata()
        assert meta["frame_ref"] == "f_abc123"

        reconstructed = MemoryItem.from_chromadb(
            doc_id="test_id", document=item.document, metadata=meta
        )
        assert reconstructed.frame_ref == "f_abc123"

    def test_frame_ref_none_by_default(self):
        item = MemoryItem(
            document="No frame",
            tier=MemoryTier.M1,
            memory_type=MemoryType.OBSERVATION,
            robot_id="test",
        )
        assert item.frame_ref is None
        meta = item.to_metadata()
        assert "frame_ref" not in meta
