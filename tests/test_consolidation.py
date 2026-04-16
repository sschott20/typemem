import time
import pytest
from typemem.consolidation import ConsolidationEngine
from typemem.plugins.base import ConsolidationPlugin
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType


class SimpleConsolidator(ConsolidationPlugin):
    """Example plugin using the tag-based pattern to track processed items."""
    def __init__(self):
        self.run_count = 0

    @property
    def name(self):
        return "simple"

    @property
    def interval_seconds(self):
        return 0.1

    def run(self, manager, llm=None):
        self.run_count += 1
        processed_tag = f"processed:{self.name}"
        unprocessed = [
            it for it in manager.get_by_tier(MemoryTier.M1)
            if processed_tag not in it.tags
        ]
        if len(unprocessed) < 2:
            return []
        summary = "[Summary] " + "; ".join(i.document for i in unprocessed)
        item = MemoryItem(
            document=summary, tier=MemoryTier.M2,
            memory_type=MemoryType.SUMMARY, robot_id="test",
        )
        mid = manager.add(item)
        for it in unprocessed:
            manager.add_tag(it.id, processed_tag)
        return [mid]


class TestConsolidationEngine:
    def test_register_and_list(self, manager):
        engine = ConsolidationEngine(manager)
        plugin = SimpleConsolidator()
        engine.register_strategy(plugin)
        assert "simple" in engine.list_strategies()

    def test_run_all(self, manager):
        engine = ConsolidationEngine(manager)
        plugin = SimpleConsolidator()
        engine.register_strategy(plugin)
        for i in range(3):
            manager.add(MemoryItem(
                document=f"observation {i}", tier=MemoryTier.M1,
                memory_type=MemoryType.OBSERVATION, robot_id="r1",
            ))
        results = engine.run_all()
        assert "simple" in results
        assert len(results["simple"]) >= 1
        assert manager.count(tier=MemoryTier.M2) >= 1

    def test_processed_index_prevents_reprocessing(self, manager):
        engine = ConsolidationEngine(manager)
        plugin = SimpleConsolidator()
        engine.register_strategy(plugin)
        for i in range(3):
            manager.add(MemoryItem(
                document=f"obs {i}", tier=MemoryTier.M1,
                memory_type=MemoryType.OBSERVATION, robot_id="r1",
            ))
        engine.run_all()
        m2_count_after_first = manager.count(tier=MemoryTier.M2)
        engine.run_all()
        assert manager.count(tier=MemoryTier.M2) == m2_count_after_first

    def test_start_and_stop(self, manager):
        engine = ConsolidationEngine(manager)
        plugin = SimpleConsolidator()
        engine.register_strategy(plugin)
        engine.start(tick_interval=0.05)
        time.sleep(0.3)
        engine.stop()
        assert plugin.run_count >= 1

    def test_expire_tiers_via_gc_plugin(self, manager):
        """Retention/expiry is now a plugin concern, not core consolidation engine."""
        from typemem.plugins.tier_retention_gc import TierRetentionGC
        old = MemoryItem(
            document="old item", tier=MemoryTier.M1,
            memory_type=MemoryType.OBSERVATION, robot_id="r1",
            timestamp=time.time() - 9999,
        )
        manager.add(old)
        assert manager.count() == 1
        gc = TierRetentionGC()
        gc.run(manager)
        assert manager.count() == 0
