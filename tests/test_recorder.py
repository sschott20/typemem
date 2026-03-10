import json
import pytest
from typemem.recorder import SessionRecorder, RecordedEvent


class TestSessionRecorder:
    def test_record_write(self, tmp_path):
        r = SessionRecorder(str(tmp_path / "events.jsonl"))
        r.record_write(tier="M1", document="saw chair", metadata={"tier": "M1"}, item_id="abc")
        assert len(r.get_events()) == 1
        assert r.get_events()[0].event == "write"

    def test_record_consolidation(self, tmp_path):
        r = SessionRecorder(str(tmp_path / "events.jsonl"))
        r.record_consolidation(strategy="m1_to_m2", source_ids=["a", "b"], result_id="c")
        ev = r.get_events()[0]
        assert ev.event == "consolidate"
        assert ev.data["strategy"] == "m1_to_m2"

    def test_record_injection(self, tmp_path):
        r = SessionRecorder(str(tmp_path / "events.jsonl"))
        r.record_injection(stage="S1", memory_ids=["x"], scores=[0.9])
        ev = r.get_events()[0]
        assert ev.event == "inject"
        assert ev.data["stage"] == "S1"

    def test_flush_and_load(self, tmp_path):
        path = str(tmp_path / "events.jsonl")
        r = SessionRecorder(path)
        r.record_write(tier="M1", document="test", metadata={}, item_id="id1")
        r.record_write(tier="M2", document="test2", metadata={}, item_id="id2")
        r.flush()
        assert len(r.get_events()) == 0
        loaded = r.load_events()
        assert len(loaded) == 2
        assert loaded[0].event == "write"

    def test_disabled_recorder(self, tmp_path):
        r = SessionRecorder(str(tmp_path / "events.jsonl"), enabled=False)
        r.record_write(tier="M1", document="test", metadata={}, item_id="id1")
        assert len(r.get_events()) == 0

    def test_auto_flush(self, tmp_path):
        path = str(tmp_path / "events.jsonl")
        r = SessionRecorder(path, flush_interval=3)
        r.record_write(tier="M1", document="a", metadata={}, item_id="1")
        r.record_write(tier="M1", document="b", metadata={}, item_id="2")
        r.record_write(tier="M1", document="c", metadata={}, item_id="3")
        loaded = r.load_events()
        assert len(loaded) == 3

    def test_replay(self, tmp_path):
        r = SessionRecorder(str(tmp_path / "events.jsonl"))
        r.record_write(tier="M1", document="a", metadata={}, item_id="1")
        r.record_write(tier="M1", document="b", metadata={}, item_id="2")
        replayed = list(r.replay())
        assert len(replayed) == 2
