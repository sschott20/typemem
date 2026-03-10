# tests/test_factory.py
import pytest
from typemem import create_memory_system
from typemem.plugins.base import ObservationPlugin


class StubPlugin(ObservationPlugin):
    @property
    def name(self):
        return "stub"
    @property
    def interval_seconds(self):
        return 1.0
    def run(self):
        return []


class TestCreateMemorySystem:
    def test_basic_creation(self, tmp_path):
        manager, engine, injector, recorder, runner, fs = create_memory_system(
            persist_dir=str(tmp_path / "chroma"),
            robot_id="r1",
        )
        assert manager is not None
        assert engine is not None
        assert injector is not None
        assert recorder is not None
        assert runner is not None
        assert fs is None

    def test_with_extra_plugins(self, tmp_path):
        plugin = StubPlugin()
        manager, engine, injector, recorder, runner, fs = create_memory_system(
            persist_dir=str(tmp_path / "chroma"),
            robot_id="r1",
            extra_plugins=[plugin],
        )
        assert "stub" in runner.list_plugins()

    def test_with_frame_store(self, tmp_path):
        manager, engine, injector, recorder, runner, fs = create_memory_system(
            persist_dir=str(tmp_path / "chroma"),
            robot_id="r1",
            frame_store_dir=str(tmp_path / "frames"),
        )
        assert fs is not None

    def test_with_recording(self, tmp_path):
        rec_path = str(tmp_path / "session.jsonl")
        manager, engine, injector, recorder, runner, fs = create_memory_system(
            persist_dir=str(tmp_path / "chroma"),
            robot_id="r1",
            recording_path=rec_path,
        )
        assert recorder is not None
