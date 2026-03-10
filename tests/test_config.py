# tests/test_config.py
import pytest
import yaml
from typemem.config import MemoryConfig, system_from_config
from typemem.injector import StageConfig
from typemem.memory_item import MemoryTier


class TestMemoryConfig:
    def test_defaults(self):
        cfg = MemoryConfig()
        assert cfg.link_top_k == 3
        assert cfg.dedup_distance_threshold == 0.05
        assert cfg.expiry_interval == 60.0

    def test_custom_values(self):
        cfg = MemoryConfig(link_top_k=5, dedup_distance_threshold=0.1)
        assert cfg.link_top_k == 5
        assert cfg.dedup_distance_threshold == 0.1


class TestSystemFromConfig:
    def test_load_from_yaml(self, tmp_path):
        config_yaml = {
            "memory": {
                "link_top_k": 5,
                "dedup_distance_threshold": 0.1,
            },
        }
        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config_yaml, f)

        persist_dir = str(tmp_path / "chroma")
        result = system_from_config(str(yaml_path), robot_id="r1", persist_dir=persist_dir)
        manager, engine, injector, recorder, runner, frame_store = result
        assert manager is not None
        assert frame_store is None  # no frame_store_dir in config

    def test_load_from_dict(self, tmp_path):
        config = {"memory": {"link_top_k": 2}}
        persist_dir = str(tmp_path / "chroma")
        result = system_from_config(config, robot_id="r1", persist_dir=persist_dir)
        assert result[0] is not None  # manager

    def test_with_frame_store(self, tmp_path):
        config = {"frame_store_dir": str(tmp_path / "frames")}
        persist_dir = str(tmp_path / "chroma")
        result = system_from_config(config, robot_id="r1", persist_dir=persist_dir)
        assert result[5] is not None  # frame_store
