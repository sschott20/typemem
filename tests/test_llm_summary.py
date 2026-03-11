"""Tests for the LLMSummaryPlugin consolidation plugin."""

import pytest
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex
from typemem.plugins.llm_summary import LLMSummaryPlugin


@pytest.fixture
def manager(tmp_path):
    return MemoryManager(persist_dir=str(tmp_path / "chroma"), robot_id="test")


@pytest.fixture
def processed_index(tmp_path):
    return ProcessedIndex(str(tmp_path / "processed.json"))


def _add_m1(manager, text, robot_id="test"):
    item = MemoryItem(
        document=text, tier=MemoryTier.M1,
        memory_type=MemoryType.OBSERVATION, robot_id=robot_id,
    )
    return manager.add(item)


def _mock_llm(prompt: str) -> str:
    return "Summary: objects observed in kitchen area"


class TestLLMSummaryPlugin:
    def test_properties(self):
        plugin = LLMSummaryPlugin(batch_size=5, interval=30.0)
        assert plugin.name == "llm_summary"
        assert plugin.source_tier == MemoryTier.M1
        assert plugin.target_tier == MemoryTier.M2
        assert plugin.interval_seconds == 30.0

    def test_no_consolidation_below_batch_size(self, manager, processed_index):
        plugin = LLMSummaryPlugin(batch_size=3)
        _add_m1(manager, "obs 1")
        _add_m1(manager, "obs 2")
        result = plugin.run(manager, llm=_mock_llm, processed_index=processed_index)
        assert result == []
        assert manager.count(tier=MemoryTier.M2) == 0

    def test_consolidates_with_llm(self, manager, processed_index):
        plugin = LLMSummaryPlugin(batch_size=3)
        _add_m1(manager, "saw a cup on counter")
        _add_m1(manager, "saw a plate on table")
        _add_m1(manager, "person entered kitchen")
        result = plugin.run(manager, llm=_mock_llm, processed_index=processed_index)
        assert len(result) == 1
        assert manager.count(tier=MemoryTier.M2) == 1
        m2_items = manager.get_by_tier(MemoryTier.M2)
        assert m2_items[0].document == "Summary: objects observed in kitchen area"
        assert m2_items[0].memory_type == MemoryType.SUMMARY

    def test_skips_when_no_llm(self, manager, processed_index):
        plugin = LLMSummaryPlugin(batch_size=2)
        _add_m1(manager, "obs 1")
        _add_m1(manager, "obs 2")
        result = plugin.run(manager, llm=None, processed_index=processed_index)
        assert result == []

    def test_idempotent_run(self, manager, processed_index):
        plugin = LLMSummaryPlugin(batch_size=2)
        _add_m1(manager, "obs A")
        _add_m1(manager, "obs B")
        result1 = plugin.run(manager, llm=_mock_llm, processed_index=processed_index)
        assert len(result1) == 1
        result2 = plugin.run(manager, llm=_mock_llm, processed_index=processed_index)
        assert result2 == []

    def test_multiple_batches(self, manager, processed_index):
        # LLM returns very different summaries so dedup doesn't collapse them
        summaries = [
            "Kitchen area: red cup and blue plate spotted on counter",
            "Warehouse zone: forklift moving pallets near loading dock",
        ]
        counter = {"n": 0}
        def unique_llm(prompt: str) -> str:
            idx = counter["n"]
            counter["n"] += 1
            return summaries[idx]
        plugin = LLMSummaryPlugin(batch_size=2)
        for i in range(5):
            _add_m1(manager, f"observation {i}")
        result = plugin.run(manager, llm=unique_llm, processed_index=processed_index)
        assert len(result) == 2
        assert manager.count(tier=MemoryTier.M2) == 2

    def test_prompt_contains_observations(self, manager, processed_index):
        prompts_seen = []
        def capturing_llm(prompt: str) -> str:
            prompts_seen.append(prompt)
            return "captured summary"
        plugin = LLMSummaryPlugin(batch_size=2)
        _add_m1(manager, "red cup on counter")
        _add_m1(manager, "blue plate on table")
        plugin.run(manager, llm=capturing_llm, processed_index=processed_index)
        assert len(prompts_seen) == 1
        assert "red cup on counter" in prompts_seen[0]
        assert "blue plate on table" in prompts_seen[0]

    def test_preserves_robot_id(self, manager, processed_index):
        plugin = LLMSummaryPlugin(batch_size=2)
        _add_m1(manager, "obs 1", robot_id="test")
        _add_m1(manager, "obs 2", robot_id="test")
        plugin.run(manager, llm=_mock_llm, processed_index=processed_index)
        m2_items = manager.get_by_tier(MemoryTier.M2)
        assert m2_items[0].robot_id == "test"
