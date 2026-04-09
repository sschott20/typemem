"""Tests for SceneObserver."""

import pytest

from typemem import events
from typemem.memory_item import MemoryTier, MemoryType
from typemem.plugins.scene_observer import SceneObserver


@pytest.fixture(autouse=True)
def reset_event_bus():
    events.reset()
    yield
    events.reset()


class TestSceneObserver:
    def test_emits_one_item_per_unique_name_waypoint(self, manager):
        obs = SceneObserver()
        obs.setup(manager, robot_id="bench")

        events.push("scene_object", {"name": "chair", "waypoint": 1, "position": [0, 0, 0]})
        events.push("scene_object", {"name": "chair", "waypoint": 2, "position": [1, 0, 0]})
        events.push("scene_object", {"name": "dog", "waypoint": 1, "position": [0, 0, 0]})

        emitted = obs.run()

        assert len(emitted) == 3
        items = manager.get_by_tier(MemoryTier.M1)
        docs = sorted(it.document for it in items)
        assert docs == [
            "Saw chair at waypoint 1",
            "Saw chair at waypoint 2",
            "Saw dog at waypoint 1",
        ]

    def test_dedupes_repeated_detections(self, manager):
        """The whole point of in-process dedup: 1000 chair-at-wp1 events → 1 item."""
        obs = SceneObserver()
        obs.setup(manager, robot_id="bench")

        for _ in range(1000):
            events.push("scene_object", {"name": "chair", "waypoint": 1, "position": [0, 0, 0]})

        emitted = obs.run()

        assert len(emitted) == 1
        assert manager.count() == 1

    def test_dedup_persists_across_run_calls(self, manager):
        obs = SceneObserver()
        obs.setup(manager, robot_id="bench")

        events.push("scene_object", {"name": "chair", "waypoint": 1})
        obs.run()
        events.push("scene_object", {"name": "chair", "waypoint": 1})
        emitted = obs.run()

        assert emitted == []
        assert manager.count() == 1

    def test_writes_m1_observation_with_waypoint(self, manager):
        obs = SceneObserver()
        obs.setup(manager, robot_id="bench")

        events.push("scene_object", {"name": "vase", "waypoint": 5})
        obs.run()

        items = manager.get_by_tier(MemoryTier.M1)
        assert len(items) == 1
        item = items[0]
        assert item.tier == MemoryTier.M1
        assert item.memory_type == MemoryType.OBSERVATION
        assert item.waypoint == 5
        assert item.source == "scene_observer"
        assert item.robot_id == "bench"

    def test_skips_events_missing_required_fields(self, manager):
        obs = SceneObserver()
        obs.setup(manager, robot_id="bench")

        events.push("scene_object", {"name": "chair"})  # no waypoint
        events.push("scene_object", {"waypoint": 1})    # no name
        events.push("scene_object", {"name": "dog", "waypoint": 2})  # ok

        obs.run()

        assert manager.count() == 1
        items = manager.get_by_tier(MemoryTier.M1)
        assert items[0].document == "Saw dog at waypoint 2"

    def test_run_before_setup_returns_empty(self):
        obs = SceneObserver()
        # No setup() call.
        assert obs.run() == []

    def test_subscribes_with_large_queue(self):
        """Default 1000-cap deque would tail-drop a 3500-event recording."""
        obs = SceneObserver(queue_size=10_000)
        # We can verify by checking the deque maxlen after setup.
        from typemem.memory_manager import MemoryManager
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            mgr = MemoryManager(persist_dir=tmp, robot_id="bench")
            obs.setup(mgr, "bench")
            assert obs._dq is not None
            assert obs._dq.maxlen == 10_000

    def test_replay_integration(self, manager, tmp_path):
        """End-to-end: write a tiny JSONL, replay it, observer should emit items."""
        import json
        rec_dir = tmp_path / "rec"
        rec_dir.mkdir()
        with open(rec_dir / "events.jsonl", "w") as f:
            for i, (name, wp) in enumerate([
                ("chair", 1), ("chair", 1), ("dog", 2), ("chair", 2),
            ]):
                f.write(json.dumps({
                    "ch": "scene_object",
                    "data": {"name": name, "waypoint": wp},
                    "ts": 100.0 + i,
                }) + "\n")

        obs = SceneObserver()
        obs.setup(manager, robot_id="bench")

        from typemem.replay import EventReplay
        replay = EventReplay(str(rec_dir))
        replay.play(speed=0)
        obs.run()

        # 3 unique (name, waypoint) pairs: (chair,1) (dog,2) (chair,2)
        assert manager.count() == 3
