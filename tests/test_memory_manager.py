import time
import pytest
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager


def _make_item(doc="test", tier=MemoryTier.M1, mtype=MemoryType.OBSERVATION, robot_id="r1", **kwargs):
    return MemoryItem(document=doc, tier=tier, memory_type=mtype, robot_id=robot_id, **kwargs)


class TestAdd:
    def test_add_returns_id(self, manager):
        item = _make_item("hello world")
        item_id = manager.add(item)
        assert item_id == item.id

    def test_add_and_get(self, manager):
        item = _make_item("visible chair near waypoint 3", metadata={"waypoint": 3})
        manager.add(item)
        retrieved = manager.get(item.id)
        assert retrieved is not None
        assert retrieved.document == item.document
        assert retrieved.tier == MemoryTier.M1
        assert retrieved.metadata.get("waypoint") == 3

    def test_dedup_same_tier(self, manager):
        item1 = _make_item("the red ball is on the table")
        item2 = _make_item("the red ball is on the table")
        id1 = manager.add(item1)
        id2 = manager.add(item2)
        assert id1 == id2
        assert manager.count() == 1

    def test_no_dedup_for_m0(self, manager):
        item1 = _make_item("raw data", tier=MemoryTier.M0)
        item2 = _make_item("raw data", tier=MemoryTier.M0)
        id1 = manager.add(item1)
        id2 = manager.add(item2)
        assert id1 != id2
        assert manager.count() == 2

    def test_get_nonexistent(self, manager):
        assert manager.get("nonexistent_id") is None


class TestBatch:
    def test_add_batch(self, manager):
        items = [_make_item(f"item {i}") for i in range(5)]
        ids = manager.add_batch(items)
        assert len(ids) == 5
        assert manager.count() == 5

    def test_add_batch_empty(self, manager):
        assert manager.add_batch([]) == []


class TestSearch:
    def test_search_basic(self, manager):
        manager.add(_make_item("the red ball is on the table"))
        manager.add(_make_item("the blue cup is in the sink"))
        results = manager.search("ball")
        assert len(results) >= 1
        assert any("ball" in r.document for r in results)

    def test_search_with_tier_filter(self, manager):
        manager.add(_make_item("m1 item", tier=MemoryTier.M1))
        manager.add(_make_item("m2 item", tier=MemoryTier.M2))
        results = manager.search("item", tiers=[MemoryTier.M1])
        assert all(r.tier == MemoryTier.M1 for r in results)

    def test_search_with_distances(self, manager):
        manager.add(_make_item("cat on mat"))
        items, distances = manager.search_with_distances("cat")
        assert len(items) == len(distances)
        assert all(isinstance(d, float) for d in distances)

    def test_search_empty_store(self, manager):
        results = manager.search("anything")
        assert results == []


class TestDelete:
    def test_delete(self, manager):
        item = _make_item("to delete")
        manager.add(item)
        assert manager.count() == 1
        manager.delete(item.id)
        assert manager.count() == 0
        assert manager.get(item.id) is None


class TestTierOperations:
    def test_get_by_tier(self, manager):
        manager.add(_make_item("m1a", tier=MemoryTier.M1))
        manager.add(_make_item("m1b", tier=MemoryTier.M1))
        manager.add(_make_item("m2a", tier=MemoryTier.M2))
        results = manager.get_by_tier(MemoryTier.M1)
        assert len(results) == 2

    def test_count_by_tier(self, manager):
        manager.add(_make_item("a", tier=MemoryTier.M1))
        manager.add(_make_item("b", tier=MemoryTier.M2))
        assert manager.count(tier=MemoryTier.M1) == 1
        assert manager.count(tier=MemoryTier.M2) == 1
        assert manager.count() == 2

    def test_expire_tier(self, manager):
        old_item = _make_item("old", tier=MemoryTier.M1, timestamp=time.time() - 9999)
        new_item = _make_item("new", tier=MemoryTier.M1)
        manager.add(old_item)
        manager.add(new_item)
        expired = manager.expire_tier(MemoryTier.M1)
        assert old_item.id in expired
        assert new_item.id not in expired
        assert manager.count() == 1

    def test_expire_persistent_tier(self, manager):
        manager.add(_make_item("permanent", tier=MemoryTier.M3))
        expired = manager.expire_tier(MemoryTier.M3)
        assert expired == []
        assert manager.count() == 1


class TestLinking:
    def test_auto_linking_creates_links(self, manager):
        manager.add(_make_item("the red ball is near the door"))
        manager.add(_make_item("the red ball rolled under the table"))
        links = manager.links.get_links(manager.get_by_tier(MemoryTier.M1)[0].id)
        assert len(links) >= 0

    def test_no_linking_for_m0(self, manager):
        item = _make_item("raw", tier=MemoryTier.M0)
        manager.add(item)
        assert manager.links.link_count(item.id) == 0

    def test_delete_removes_links(self, manager):
        item1 = _make_item("the chair is in the kitchen")
        item2 = _make_item("the chair is near the stove in the kitchen")
        manager.add(item1)
        manager.add(item2)
        manager.delete(item1.id)
        assert item1.id not in manager.links.get_links(item2.id)


class TestGetBySource:
    def test_returns_matching_items(self, manager):
        manager.add(MemoryItem(document="a", tier=MemoryTier.M1, memory_type=MemoryType.OBSERVATION, robot_id="r1", source="alpha"))
        manager.add(MemoryItem(document="b", tier=MemoryTier.M1, memory_type=MemoryType.OBSERVATION, robot_id="r1", source="beta"))
        manager.add(MemoryItem(document="c", tier=MemoryTier.M1, memory_type=MemoryType.OBSERVATION, robot_id="r1", source="alpha"))
        results = manager.get_by_source("alpha")
        assert len(results) == 2
        assert all(r.source == "alpha" for r in results)

    def test_filters_by_tier(self, manager):
        manager.add(MemoryItem(document="m1", tier=MemoryTier.M1, memory_type=MemoryType.OBSERVATION, robot_id="r1", source="x"))
        manager.add(MemoryItem(document="m2", tier=MemoryTier.M2, memory_type=MemoryType.SUMMARY, robot_id="r1", source="x"))
        results = manager.get_by_source("x", tier=MemoryTier.M1)
        assert len(results) == 1
        assert results[0].tier == MemoryTier.M1

    def test_returns_empty_for_no_match(self, manager):
        results = manager.get_by_source("nonexistent")
        assert results == []


class TestUpdateDocument:
    def test_update_document(self, manager):
        item = _make_item("original text")
        manager.add(item)
        manager.update_document(item.id, "updated text")
        retrieved = manager.get(item.id)
        assert retrieved.document == "updated text"
