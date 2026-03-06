"""Tests for MemorySystem registration and manual mode."""
import time

import pytest
from typemem.system import MemorySystem


@pytest.fixture
def system(store):
    """Fresh MemorySystem per test, ensuring stop() is called on teardown."""
    sys = MemorySystem(store)
    yield sys
    sys.stop()


def test_register_and_observe(store):
    """Register observation fn that writes objects from raw_data to store."""
    system = MemorySystem(store)

    def observe_objects(raw_data, s):
        ids = []
        for obj in raw_data.get("objects", []):
            mid = s.add(obj, metadata={"tier": "episodic"})
            ids.append(mid)
        return ids

    system.add_observation("objects", observe_objects, interval=1.0)
    ids = system.observe({"objects": ["red cup on table", "blue plate near sink"]})
    assert len(ids) == 2
    assert store.count() == 2
    for mid in ids:
        entry = store.get(mid)
        assert entry is not None
        assert entry.metadata["tier"] == "episodic"


def test_register_and_consolidate(store):
    """Seed store with items, register consolidation fn that summarizes."""
    store.add("the red cup is on the table", metadata={"tier": "episodic"})
    store.add("the red cup was moved to the shelf", metadata={"tier": "episodic"})
    store.add("the red cup fell off the shelf", metadata={"tier": "episodic"})

    system = MemorySystem(store)

    def consolidate_summary(s):
        entries = s.get_all(filters={"tier": "episodic"})
        if len(entries) >= 3:
            summary = "Summary: " + "; ".join(e.text for e in entries)
            mid = s.add(summary, metadata={"tier": "semantic"})
            return [mid]
        return []

    system.add_consolidation("summarize", consolidate_summary, interval=5.0)
    ids = system.consolidate()
    assert len(ids) == 1
    summary_entry = store.get(ids[0])
    assert summary_entry is not None
    assert summary_entry.metadata["tier"] == "semantic"
    assert "Summary:" in summary_entry.text


def test_register_and_inject(store):
    """Seed store, register injection fn that searches and formats context."""
    store.add("the red cup is on the table", metadata={"tier": "episodic"})
    store.add("the robot arm is at home position", metadata={"tier": "episodic"})
    store.add("the blue plate is near the sink", metadata={"tier": "episodic"})

    system = MemorySystem(store)

    def inject_context(query, s, token_budget):
        results = s.search(query, n=3)
        lines = []
        budget_left = token_budget
        for r in results:
            line = f"- {r.entry.text}"
            # Rough token estimate: 1 token per 4 chars
            cost = len(line) // 4
            if cost > budget_left:
                break
            lines.append(line)
            budget_left -= cost
        return "\n".join(lines)

    system.add_injection("context", inject_context)
    result = system.inject("context", "cup on table", token_budget=200)
    assert "cup" in result or "table" in result
    assert len(result) > 0


def test_inject_unknown_name_raises(store):
    """KeyError when injection name doesn't exist."""
    system = MemorySystem(store)
    with pytest.raises(KeyError, match="No injection function named"):
        system.inject("nonexistent", "query", token_budget=100)


def test_multiple_observation_fns(store):
    """Register 2 observation fns, verify both run on observe()."""
    system = MemorySystem(store)

    def observe_objects(raw_data, s):
        ids = []
        for obj in raw_data.get("objects", []):
            mid = s.add(obj, metadata={"source": "vision"})
            ids.append(mid)
        return ids

    def observe_speech(raw_data, s):
        ids = []
        for utterance in raw_data.get("speech", []):
            mid = s.add(utterance, metadata={"source": "audio"})
            ids.append(mid)
        return ids

    system.add_observation("objects", observe_objects, interval=1.0)
    system.add_observation("speech", observe_speech, interval=1.0)

    ids = system.observe({
        "objects": ["red cup", "blue plate"],
        "speech": ["pick up the cup"],
    })
    assert len(ids) == 3
    assert store.count() == 3
    vision_items = store.get_all(filters={"source": "vision"})
    audio_items = store.get_all(filters={"source": "audio"})
    assert len(vision_items) == 2
    assert len(audio_items) == 1


def test_multiple_consolidation_fns(store):
    """Register 2 consolidation fns, verify both run on consolidate()."""
    store.add("event A happened", metadata={"tier": "episodic"})
    store.add("event B happened", metadata={"tier": "episodic"})

    system = MemorySystem(store)

    def consolidate_count(s):
        count = s.count()
        mid = s.add(f"Total memories: {count}", metadata={"tier": "meta"})
        return [mid]

    def consolidate_marker(s):
        mid = s.add("Consolidation ran", metadata={"tier": "marker"})
        return [mid]

    system.add_consolidation("count", consolidate_count, interval=5.0)
    system.add_consolidation("marker", consolidate_marker, interval=5.0)

    ids = system.consolidate()
    assert len(ids) == 2
    meta_items = store.get_all(filters={"tier": "meta"})
    marker_items = store.get_all(filters={"tier": "marker"})
    assert len(meta_items) == 1
    assert len(marker_items) == 1


def test_background_observation(system, store):
    """Background mode calls observation functions on timer."""
    call_count = {"n": 0}

    def obs_counter(raw, s):
        call_count["n"] += 1
        return [s.add(f"obs {call_count['n']}")]

    system.add_observation("counter", obs_counter, interval=0.2)
    system.start(data_source=lambda: {"tick": True})
    time.sleep(1.5)
    system.stop()
    assert call_count["n"] >= 2


def test_background_consolidation(system, store):
    """Background mode calls consolidation functions on timer."""
    store.add("seed data")
    call_count = {"n": 0}

    def cons_counter(s):
        call_count["n"] += 1
        return []

    system.add_consolidation("counter", cons_counter, interval=0.3)
    system.start(data_source=lambda: {})
    time.sleep(1.5)
    system.stop()
    assert call_count["n"] >= 2


def test_start_stop_idempotent(system):
    """Stopping without starting doesn't crash."""
    system.stop()  # should not raise
