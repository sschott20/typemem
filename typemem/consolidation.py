import logging
import os
import time
import threading
from typing import Dict, List, Optional

from typemem.memory_item import MemoryTier
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    """Runs consolidation strategies on a schedule."""

    def __init__(self, manager: MemoryManager, expiry_interval: float = 60.0):
        self._manager = manager
        self._strategies: Dict[str, object] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._recorder = None
        self._expiry_interval = expiry_interval
        self._last_expiry: float = 0.0
        self._processed_index = ProcessedIndex(
            os.path.join(manager._persist_dir, "processed.json")
        )

    def set_recorder(self, recorder):
        self._recorder = recorder

    @property
    def processed_index(self) -> ProcessedIndex:
        return self._processed_index

    def register_strategy(self, strategy):
        self._strategies[strategy.name] = strategy

    def list_strategies(self) -> List[str]:
        return list(self._strategies.keys())

    def run_all(self, llm=None) -> Dict[str, List[str]]:
        results = {}
        for name, strategy in self._strategies.items():
            new_ids = strategy.run(self._manager, llm=llm, processed_index=self._processed_index)
            results[name] = new_ids
            if self._recorder and new_ids:
                for nid in new_ids:
                    self._recorder.record_consolidation(
                        strategy=name, source_ids=[], result_id=nid,
                    )
        self._maybe_expire()
        return results

    def start(self, llm=None, tick_interval: float = 1.0):
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, args=(llm, tick_interval), daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run_loop(self, llm, tick_interval: float):
        last_run: Dict[str, float] = {}
        while self._running:
            now = time.time()
            for name, strategy in self._strategies.items():
                last = last_run.get(name, 0)
                if now - last >= strategy.interval_seconds:
                    try:
                        strategy.run(self._manager, llm=llm, processed_index=self._processed_index)
                    except Exception as e:
                        logger.error("Strategy '%s' failed: %s", name, e)
                    last_run[name] = now
            self._maybe_expire()
            time.sleep(tick_interval)

    def save(self):
        self._processed_index.save()

    def _maybe_expire(self):
        now = time.time()
        if now - self._last_expiry >= self._expiry_interval:
            for tier in [MemoryTier.M0, MemoryTier.M1, MemoryTier.M2]:
                self._manager.expire_tier(tier)
            self._last_expiry = now

            all_ids = set(self._manager._collection.get(include=[])["ids"])
            self._processed_index.prune(all_ids)
            self._processed_index.save()
