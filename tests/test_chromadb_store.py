"""Tests for ChromaDBStore implementation."""
from typemem.types import MemoryEntry


def test_add_and_get(store):
    mid = store.add("the cup is on the table")
    entry = store.get(mid)
    assert entry is not None
    assert entry.id == mid
    assert entry.text == "the cup is on the table"


def test_add_with_metadata(store):
    mid = store.add("robot arm at position 0,0,1", metadata={"tier": "episodic", "source": "vision"})
    entry = store.get(mid)
    assert entry is not None
    assert entry.metadata["tier"] == "episodic"
    assert entry.metadata["source"] == "vision"


def test_add_with_custom_id(store):
    mid = store.add("custom id memory", id="my_custom_id")
    assert mid == "my_custom_id"
    entry = store.get("my_custom_id")
    assert entry is not None
    assert entry.text == "custom id memory"


def test_search(store):
    store.add("the cup is on the table")
    store.add("the robot arm is at home position")
    store.add("the mug was placed near the plate")
    results = store.search("cup on table", n=3)
    assert len(results) == 3
    # The most relevant result should be about the cup on the table
    assert "cup" in results[0].entry.text or "mug" in results[0].entry.text
    # Distances should be sorted ascending (most relevant first)
    assert results[0].distance <= results[1].distance


def test_search_with_filters(store):
    store.add("episodic memory about cup", metadata={"tier": "episodic"})
    store.add("semantic memory about cups", metadata={"tier": "semantic"})
    store.add("another episodic memory", metadata={"tier": "episodic"})
    results = store.search("cup", n=10, filters={"tier": "episodic"})
    assert len(results) == 2
    for r in results:
        assert r.entry.metadata["tier"] == "episodic"


def test_delete(store):
    mid = store.add("memory to delete")
    assert store.get(mid) is not None
    store.delete(mid)
    assert store.get(mid) is None


def test_update_text(store):
    mid = store.add("original text")
    store.update(mid, text="updated text")
    entry = store.get(mid)
    assert entry is not None
    assert entry.text == "updated text"


def test_update_metadata(store):
    mid = store.add("some memory", metadata={"tier": "episodic"})
    store.update(mid, metadata={"tier": "semantic", "importance": "high"})
    entry = store.get(mid)
    assert entry is not None
    assert entry.metadata["tier"] == "semantic"
    assert entry.metadata["importance"] == "high"


def test_get_all(store):
    store.add("memory one", metadata={"tier": "episodic"})
    store.add("memory two", metadata={"tier": "semantic"})
    store.add("memory three", metadata={"tier": "episodic"})
    all_memories = store.get_all()
    assert len(all_memories) == 3
    # With filters
    episodic = store.get_all(filters={"tier": "episodic"})
    assert len(episodic) == 2
    for m in episodic:
        assert m.metadata["tier"] == "episodic"


def test_count(store):
    assert store.count() == 0
    store.add("memory one")
    store.add("memory two")
    assert store.count() == 2


def test_count_with_filters(store):
    store.add("memory one", metadata={"tier": "episodic"})
    store.add("memory two", metadata={"tier": "semantic"})
    store.add("memory three", metadata={"tier": "episodic"})
    assert store.count(filters={"tier": "episodic"}) == 2
    assert store.count(filters={"tier": "semantic"}) == 1


def test_get_nonexistent(store):
    result = store.get("nonexistent_id")
    assert result is None


def test_search_empty_store(store):
    results = store.search("anything")
    assert results == []
