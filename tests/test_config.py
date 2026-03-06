import pytest
from pathlib import Path
from typemem.config import load_config, system_from_config

EXAMPLE_CONFIG = Path(__file__).parent.parent / "configs" / "examples" / "home_assistant.yaml"


def test_load_config():
    config = load_config(EXAMPLE_CONFIG)
    assert config["personality"]
    assert len(config["observation"]) >= 1
    assert len(config["consolidation"]) >= 1
    assert len(config["injection"]) >= 1


def test_system_from_config(store):
    system = system_from_config(EXAMPLE_CONFIG, store)
    ids = system.observe({"objects": [{"name": "cup", "position": [1, 2, 3]}]})
    assert len(ids) == 1
    assert store.count() == 1


def test_system_from_config_consolidation(store):
    system = system_from_config(EXAMPLE_CONFIG, store)
    for i in range(5):
        store.add(f"object {i}", metadata={"source": "scene_objects"})
    ids = system.consolidate()
    assert len(ids) >= 1


def test_system_from_config_injection(store):
    system = system_from_config(EXAMPLE_CONFIG, store)
    store.add("Red cup on kitchen table")
    context = system.inject("task_relevant", "where is the cup?", token_budget=500)
    assert "cup" in context.lower()


def test_load_config_from_dict(store):
    config_dict = {
        "personality": "test robot",
        "observation": [{
            "name": "simple",
            "interval": 1.0,
            "code": "def fn(raw_data, store):\n    return [store.add(raw_data.get('text', ''))]",
        }],
        "consolidation": [],
        "injection": [{
            "name": "dump",
            "code": "def fn(query, store, token_budget):\n    return '\\n'.join(e.text for e in store.get_all())",
        }],
    }
    system = system_from_config(config_dict, store)
    system.observe({"text": "hello"})
    assert store.count() == 1
