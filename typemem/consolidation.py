import logging
import os
import time
import threading
from typing import Dict, List, Optional

from typemem.memory_item import MemoryTier
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex
from typemem import events

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    """Runs consolidation strategies on a schedule."""

    def __init__(self, manager: MemoryManager, expiry_interval: float = 60.0):
        self._manager = manager
        self._strategies: Dict[str, object] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._recorder = None
        self._on_consolidation = None  # callback: (strategy_name, new_ids) -> None
        self._obs_runner = None
        self._expiry_interval = expiry_interval
        self._last_expiry: float = 0.0
        self._processed_index = ProcessedIndex(
            os.path.join(manager.persist_dir, "processed.json")
        )
        # Wake signal: set by event bus when new data arrives
        self._wake = threading.Event()
        self._event_queues: List = []

    def set_recorder(self, recorder):
        self._recorder = recorder

    def set_observation_runner(self, runner):
        """Set reference to observation runner for pre-consolidation flush."""
        self._obs_runner = runner

    def set_on_consolidation(self, callback):
        """Set a callback invoked after each strategy run: callback(name, new_ids, source_ids)."""
        self._on_consolidation = callback

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
            before = self._processed_index.get_processed_set(name)
            new_ids = strategy.run(self._manager, llm=llm, processed_index=self._processed_index)
            results[name] = new_ids
            if new_ids:
                source_ids = list(self._processed_index.get_processed_set(name) - before)
                if self._recorder:
                    for nid in new_ids:
                        self._recorder.record_consolidation(
                            strategy=name, source_ids=source_ids, result_id=nid,
                        )
                if self._on_consolidation:
                    try:
                        self._on_consolidation(name, new_ids, source_ids)
                    except Exception:
                        pass
        self._maybe_expire()
        return results

    def start(self, llm=None, tick_interval: float = 1.0,
              watch_channels: tuple = ("user_instruction", "task_lifecycle", "action")):
        # Subscribe to event channels so we wake immediately on new data
        for ch in watch_channels:
            dq = events.subscribe(ch)
            self._event_queues.append(dq)
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, args=(llm, tick_interval), daemon=True,
        )
        self._thread.start()

    def wake(self):
        """Signal the run loop to check strategies immediately."""
        self._wake.set()

    def stop(self):
        self._running = False
        self._wake.set()  # unblock the wait so thread exits promptly
        if self._thread:
            self._thread.join(timeout=5.0)

    def _has_new_events(self) -> bool:
        """Check if any watched channel has new data (drain without processing)."""
        for dq in self._event_queues:
            if dq:
                events.drain(dq)
                return True
        return False

    def _run_loop(self, llm, tick_interval: float):
        last_run: Dict[str, float] = {}
        while self._running:
            # Wait for either tick timeout or wake signal from event bus
            self._wake.wait(timeout=tick_interval)
            self._wake.clear()

            has_new = self._has_new_events()
            now = time.time()

            # Flush observation plugins so M1 is up-to-date before consolidation reads
            if has_new and self._obs_runner is not None:
                try:
                    self._obs_runner.flush()
                except Exception as e:
                    logger.error("Observation flush failed: %s", e)

            for name, strategy in self._strategies.items():
                last = last_run.get(name, 0)
                elapsed = now - last
                # Run if interval elapsed, or if new events arrived and
                # a minimum cooldown (1s) has passed (observation plugins
                # are flushed above so M1 is already up-to-date)
                if elapsed >= strategy.interval_seconds or (has_new and elapsed >= 1.0):
                    try:
                        before = self._processed_index.get_processed_set(name)
                        new_ids = strategy.run(self._manager, llm=llm, processed_index=self._processed_index)
                        if new_ids:
                            source_ids = list(self._processed_index.get_processed_set(name) - before)
                            if self._recorder:
                                for nid in new_ids:
                                    self._recorder.record_consolidation(
                                        strategy=name, source_ids=source_ids, result_id=nid,
                                    )
                            if self._on_consolidation:
                                try:
                                    self._on_consolidation(name, new_ids, source_ids)
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.error("Strategy '%s' failed: %s", name, e)
                    last_run[name] = now
            self._maybe_expire()

    def save(self):
        self._processed_index.save()

    def _maybe_expire(self):
        now = time.time()
        if now - self._last_expiry >= self._expiry_interval:
            for tier in [MemoryTier.M0, MemoryTier.M1, MemoryTier.M2]:
                self._manager.expire_tier(tier)
            self._last_expiry = now

            all_ids = self._manager.all_ids()
            self._processed_index.prune(all_ids)
            self._processed_index.save()
