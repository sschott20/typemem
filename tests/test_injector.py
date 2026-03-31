import time
import pytest
from typemem.injector import MemoryInjector, StageConfig
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager


class TestMemoryInjector:
    def test_inject_empty_store(self, manager):
        inj = MemoryInjector(manager)
        result = inj.inject("S1", "anything")
        assert result == ""

    def test_inject_returns_context(self, manager):
        manager.add(MemoryItem(
            document="the red ball is on the table",
            tier=MemoryTier.M1, memory_type=MemoryType.OBSERVATION, robot_id="r1",
        ))
        inj = MemoryInjector(manager)
        result = inj.inject("S1", "where is the ball")
        assert "ball" in result

    def test_inject_respects_tier_filter(self, manager):
        manager.add(MemoryItem(
            document="m1 observation", tier=MemoryTier.M1,
            memory_type=MemoryType.OBSERVATION, robot_id="r1",
        ))
        manager.add(MemoryItem(
            document="m3 knowledge", tier=MemoryTier.M3,
            memory_type=MemoryType.LESSON, robot_id="r1",
        ))
        inj = MemoryInjector(manager)
        result = inj.inject("S1", "observation")
        assert "[M1]" in result

    def test_inject_unknown_stage(self, manager):
        inj = MemoryInjector(manager)
        result = inj.inject("UNKNOWN", "query")
        assert result == ""

    def test_set_stage_config(self, manager):
        inj = MemoryInjector(manager)
        custom = StageConfig(tiers=[MemoryTier.M3], max_tokens=100, n_results=5, recency_weight=0.0)
        inj.set_stage_config("custom", custom)
        manager.add(MemoryItem(
            document="persistent knowledge", tier=MemoryTier.M3,
            memory_type=MemoryType.LESSON, robot_id="r1",
        ))
        result = inj.inject("custom", "knowledge")
        assert "persistent" in result

    def test_token_budget_limits_output(self, manager):
        for i in range(20):
            manager.add(MemoryItem(
                document=f"observation number {i} with some padding text to consume tokens",
                tier=MemoryTier.M1, memory_type=MemoryType.OBSERVATION, robot_id="r1",
            ))
        inj = MemoryInjector(manager)
        result = inj.inject("S1", "observation")
        approx_tokens = len(result) // 4
        assert approx_tokens <= 500

    def test_cache_returns_same_result(self, manager):
        manager.add(MemoryItem(
            document="cached item", tier=MemoryTier.M1,
            memory_type=MemoryType.OBSERVATION, robot_id="r1",
        ))
        inj = MemoryInjector(manager, cache_ttl=10.0)
        r1 = inj.inject("S1", "cached")
        r2 = inj.inject("S1", "cached")
        assert r1 == r2

    def test_recorder_integration(self, manager, tmp_path):
        from typemem.recorder import SessionRecorder
        rec = SessionRecorder(str(tmp_path / "events.jsonl"))
        manager.add(MemoryItem(
            document="recorded item", tier=MemoryTier.M1,
            memory_type=MemoryType.OBSERVATION, robot_id="r1",
        ))
        inj = MemoryInjector(manager)
        inj.set_recorder(rec)
        inj.inject("S1", "recorded")
        events = rec.get_events()
        inject_events = [e for e in events if e.event == "inject"]
        assert len(inject_events) >= 1


class TestPinnedInjection:
    def test_pinned_source_prepended(self, tmp_path):
        manager = MemoryManager(persist_dir=str(tmp_path), robot_id="test")
        manager.add(MemoryItem(document="situation: person in room", tier=MemoryTier.M2, memory_type=MemoryType.SUMMARY, robot_id="test", source="situation_summary"))
        manager.add(MemoryItem(document="some other memory", tier=MemoryTier.M1, memory_type=MemoryType.OBSERVATION, robot_id="test", source="obs"))

        injector = MemoryInjector(manager, cache_ttl=0)
        injector.set_stage_config("S1", StageConfig(
            tiers=[MemoryTier.M1, MemoryTier.M2],
            max_tokens=500,
            pinned_sources=["situation_summary"],
        ))
        result = injector.inject("S1", query="what is happening")
        # Pinned item should appear first
        lines = result.strip().split("\n")
        assert "situation: person in room" in lines[0]

    def test_empty_pinned_sources_is_noop(self, tmp_path):
        manager = MemoryManager(persist_dir=str(tmp_path), robot_id="test")
        manager.add(MemoryItem(document="hello world", tier=MemoryTier.M1, memory_type=MemoryType.OBSERVATION, robot_id="test"))

        injector = MemoryInjector(manager, cache_ttl=0)
        injector.set_stage_config("S1", StageConfig(
            tiers=[MemoryTier.M1],
            max_tokens=500,
            pinned_sources=[],
        ))
        result = injector.inject("S1", query="hello")
        assert "hello" in result

    def test_pinned_first_item_always_included(self, tmp_path):
        manager = MemoryManager(persist_dir=str(tmp_path), robot_id="test")
        long_doc = "x " * 200
        manager.add(MemoryItem(document=long_doc, tier=MemoryTier.M2, memory_type=MemoryType.SUMMARY, robot_id="test", source="pinned_src"))

        injector = MemoryInjector(manager, cache_ttl=0)
        injector.set_stage_config("S1", StageConfig(
            tiers=[MemoryTier.M1, MemoryTier.M2],
            max_tokens=10,  # tiny budget
            pinned_sources=["pinned_src"],
        ))
        result = injector.inject("S1", query="anything")
        # First pinned item always included even if it exceeds budget
        assert "x x" in result
