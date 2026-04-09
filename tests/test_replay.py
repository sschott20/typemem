"""Tests for the event replay module."""

import json
import os
import time

import numpy as np
import pytest

from typemem import events
from typemem.frame_store import FrameStore
from typemem.replay import EventReplay, RawEvent, load_events


@pytest.fixture(autouse=True)
def reset_event_bus():
    events.reset()
    yield
    events.reset()


def _write_events(path, evts):
    """Helper: write a list of RawEvent to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for ev in evts:
            f.write(json.dumps(ev.to_dict()) + "\n")


class TestRawEvent:
    def test_roundtrip(self):
        ev = RawEvent("action", {"skill": "walk"}, 123.0)
        d = ev.to_dict()
        ev2 = RawEvent.from_dict(d)
        assert ev2.channel == "action"
        assert ev2.data == {"skill": "walk"}
        assert ev2.timestamp == 123.0


class TestLoadEvents:
    def test_loads_sorted_by_timestamp(self, tmp_path):
        path = str(tmp_path / "events.jsonl")
        evts = [
            RawEvent("b", {"x": 2}, 200.0),
            RawEvent("a", {"x": 1}, 100.0),
            RawEvent("c", {"x": 3}, 150.0),
        ]
        _write_events(path, evts)

        loaded = load_events(path)
        assert len(loaded) == 3
        assert [e.timestamp for e in loaded] == [100.0, 150.0, 200.0]


class TestEventReplay:
    def _make_recording(self, tmp_path, evts, frames=None):
        """Create a recording directory with events.jsonl and optional frames."""
        rec_dir = str(tmp_path / "recording")
        os.makedirs(rec_dir, exist_ok=True)
        _write_events(os.path.join(rec_dir, "events.jsonl"), evts)

        if frames:
            frames_dir = os.path.join(rec_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            for frame_id, img in frames.items():
                from typemem.frame_store import _encode_jpeg
                with open(os.path.join(frames_dir, f"{frame_id}.jpg"), "wb") as f:
                    f.write(_encode_jpeg(img))

        return rec_dir

    def test_play_instant_pushes_all_events(self, tmp_path):
        evts = [
            RawEvent("action", {"skill": "walk"}, 100.0),
            RawEvent("action", {"skill": "turn"}, 101.0),
            RawEvent("scene_object", {"name": "chair"}, 102.0),
        ]
        rec_dir = self._make_recording(tmp_path, evts)

        dq_action = events.subscribe("action")
        dq_scene = events.subscribe("scene_object")

        replay = EventReplay(rec_dir)
        count = replay.play(speed=0)

        assert count == 3
        assert len(dq_action) == 2
        assert len(dq_scene) == 1
        assert dq_action[0]["skill"] == "walk"

    def test_play_filters_channels(self, tmp_path):
        evts = [
            RawEvent("action", {"skill": "walk"}, 100.0),
            RawEvent("scene_object", {"name": "chair"}, 101.0),
        ]
        rec_dir = self._make_recording(tmp_path, evts)

        dq_action = events.subscribe("action")
        dq_scene = events.subscribe("scene_object")

        replay = EventReplay(rec_dir)
        count = replay.play(speed=0, channels=["action"])

        assert count == 1
        assert len(dq_action) == 1
        assert len(dq_scene) == 0

    def test_duration_property(self, tmp_path):
        evts = [
            RawEvent("a", {}, 100.0),
            RawEvent("b", {}, 130.0),
        ]
        rec_dir = self._make_recording(tmp_path, evts)

        replay = EventReplay(rec_dir)
        assert replay.duration == 30.0

    def test_channels_property(self, tmp_path):
        evts = [
            RawEvent("action", {}, 1.0),
            RawEvent("frame", {}, 2.0),
            RawEvent("action", {}, 3.0),
        ]
        rec_dir = self._make_recording(tmp_path, evts)

        replay = EventReplay(rec_dir)
        assert replay.channels == ["action", "frame"]

    def test_iter_events(self, tmp_path):
        evts = [
            RawEvent("a", {"x": 1}, 100.0),
            RawEvent("b", {"x": 2}, 200.0),
        ]
        rec_dir = self._make_recording(tmp_path, evts)

        replay = EventReplay(rec_dir)
        collected = list(replay.iter_events())
        assert len(collected) == 2
        assert collected[0].channel == "a"

    def test_iter_events_with_channel_filter(self, tmp_path):
        evts = [
            RawEvent("a", {}, 1.0),
            RawEvent("b", {}, 2.0),
        ]
        rec_dir = self._make_recording(tmp_path, evts)

        replay = EventReplay(rec_dir)
        collected = list(replay.iter_events(channels=["b"]))
        assert len(collected) == 1
        assert collected[0].channel == "b"

    @pytest.mark.skipif(
        not any(
            __import__("importlib").util.find_spec(m) for m in ("cv2", "PIL")
        ),
        reason="requires cv2 or PIL for JPEG encoding",
    )
    def test_frame_loaded_into_frame_store(self, tmp_path):
        frame_img = np.zeros((48, 64, 3), dtype=np.uint8)
        frame_img[10, 10] = [255, 0, 0]

        evts = [
            RawEvent("frame", {"frame_id": "f_test123", "timestamp": 100.0}, 100.0),
        ]
        rec_dir = self._make_recording(tmp_path, evts, frames={"f_test123": frame_img})

        fs = FrameStore(str(tmp_path / "live_frames"))
        replay = EventReplay(rec_dir, frame_store=fs)
        replay.play(speed=0)

        loaded = fs.load("f_test123")
        assert loaded is not None
        assert loaded.shape[0] == 48
        assert loaded.shape[1] == 64

    def test_missing_recording_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            EventReplay(str(tmp_path / "nonexistent"))

    def test_play_realtime_respects_timing(self, tmp_path):
        """With speed=1, playback should take roughly the recording duration."""
        evts = [
            RawEvent("a", {}, 100.0),
            RawEvent("b", {}, 100.2),  # 0.2s gap
        ]
        rec_dir = self._make_recording(tmp_path, evts)

        dq = events.subscribe("a")
        events.subscribe("b")

        replay = EventReplay(rec_dir)
        t0 = time.monotonic()
        replay.play(speed=1.0)
        elapsed = time.monotonic() - t0

        assert elapsed >= 0.15  # allow some slack

    def test_play_fast_forward(self, tmp_path):
        """With speed=10, 2s recording should take ~0.2s."""
        evts = [
            RawEvent("a", {}, 100.0),
            RawEvent("b", {}, 102.0),  # 2s gap
        ]
        rec_dir = self._make_recording(tmp_path, evts)
        events.subscribe("a")
        events.subscribe("b")

        replay = EventReplay(rec_dir)
        t0 = time.monotonic()
        replay.play(speed=10.0)
        elapsed = time.monotonic() - t0

        assert elapsed < 1.0  # should be ~0.2s
