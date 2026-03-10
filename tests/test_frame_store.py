import time
import numpy as np
import pytest
from typemem.frame_store import FrameStore


class TestFrameStore:
    def test_store_and_load(self, tmp_path):
        fs = FrameStore(str(tmp_path / "frames"))
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame_id = fs.store(frame, timestamp=time.time())
        assert frame_id.startswith("f_")
        loaded = fs.load(frame_id)
        assert loaded is not None
        assert loaded.shape[0] == 480
        assert loaded.shape[1] == 640

    def test_load_nonexistent(self, tmp_path):
        fs = FrameStore(str(tmp_path / "frames"))
        assert fs.load("f_nonexistent") is None

    def test_cleanup_expired(self, tmp_path):
        fs = FrameStore(str(tmp_path / "frames"), default_ttl=1.0)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        fid = fs.store(frame, timestamp=time.time() - 10.0)
        removed = fs.cleanup()
        assert removed >= 1
        assert fs.load(fid) is None

    def test_cleanup_keeps_recent(self, tmp_path):
        fs = FrameStore(str(tmp_path / "frames"), default_ttl=600.0)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        fid = fs.store(frame, timestamp=time.time())
        removed = fs.cleanup()
        assert removed == 0
        assert fs.load(fid) is not None

    def test_waypoint_metadata(self, tmp_path):
        fs = FrameStore(str(tmp_path / "frames"))
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        fid = fs.store(frame, timestamp=time.time(), waypoint=5)
        assert fid.startswith("f_")

    def test_custom_max_age(self, tmp_path):
        fs = FrameStore(str(tmp_path / "frames"), default_ttl=600.0)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        fid = fs.store(frame, timestamp=time.time() - 5.0)
        removed = fs.cleanup(max_age=2.0)
        assert removed >= 1
