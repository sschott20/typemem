import pytest
from unittest.mock import patch, MagicMock
from typemem.generate import generate_config, generate_functions, generate_full


MOCK_CONFIG_RESPONSE = """
observation:
  - name: scene_tracker
    interval: 1.0
    description: "Track objects in the scene"
  - name: action_logger
    interval: 5.0
    description: "Log action outcomes"

consolidation:
  - name: summarize
    interval: 30.0
    description: "Summarize recent observations"

injection:
  - name: relevant
    description: "Find task-relevant memories"
"""

MOCK_CODE_RESPONSE = '''def fn(raw_data, store):
    ids = []
    for obj in raw_data.get("objects", []):
        mid = store.add(f"Saw {obj['name']}")
        ids.append(mid)
    return ids'''


class TestGenerateConfig:
    @patch("typemem.generate._llm_call")
    def test_generates_valid_config(self, mock_llm):
        mock_llm.return_value = MOCK_CONFIG_RESPONSE
        config = generate_config("A patrol robot that monitors a building")
        assert "observation" in config
        assert "consolidation" in config
        assert "injection" in config
        assert len(config["observation"]) >= 1

    @patch("typemem.generate._llm_call")
    def test_config_has_required_fields(self, mock_llm):
        mock_llm.return_value = MOCK_CONFIG_RESPONSE
        config = generate_config("A home assistant robot")
        for obs in config["observation"]:
            assert "name" in obs
            assert "interval" in obs
            assert "description" in obs


class TestGenerateFunctions:
    @patch("typemem.generate._llm_call")
    def test_generates_code_for_observation(self, mock_llm):
        mock_llm.return_value = MOCK_CODE_RESPONSE
        fn_def = {"name": "scene_tracker", "interval": 1.0, "description": "Track objects"}
        result = generate_functions("observation", fn_def, "A patrol robot")
        assert "code" in result
        assert "def fn(" in result["code"]

    @patch("typemem.generate._llm_call")
    def test_generated_code_is_executable(self, mock_llm):
        mock_llm.return_value = MOCK_CODE_RESPONSE
        fn_def = {"name": "scene_tracker", "interval": 1.0, "description": "Track objects"}
        result = generate_functions("observation", fn_def, "A patrol robot")
        namespace = {}
        exec(result["code"], namespace)
        assert "fn" in namespace


class TestGenerateFull:
    @patch("typemem.generate._llm_call")
    def test_generate_full_produces_complete_config(self, mock_llm):
        # First call returns config YAML, subsequent calls return code
        mock_llm.side_effect = [MOCK_CONFIG_RESPONSE] + [MOCK_CODE_RESPONSE] * 4
        config = generate_full("A patrol robot")
        assert "personality" in config
        # All functions should have code
        for obs in config["observation"]:
            assert "code" in obs
        for cons in config["consolidation"]:
            assert "code" in cons
        for inj in config["injection"]:
            assert "code" in inj
