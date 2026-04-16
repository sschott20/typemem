import logging
import time
import threading
from typing import Dict, List, Optional

from typemem.memory_item import MemoryTier
from typemem.memory_manager import MemoryManager
from typemem import events

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    """Runs consolidation strategies on a schedule.

    Plugins communicate state via tags on memory items, not via a shared
    processed-index. The engine just schedules and invokes plugin.run(manager, llm).
    """

    def __init__(self, manager: MemoryManager):
        self._manager = manager
        self._strategies: Dict[str, object] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._recorder = None
        self._on_consolidation = None  # callback: (strategy_name, new_ids, source_ids) -> None
        self._obs_runner = None
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

    def register_strategy(self, strategy):
        self._strategies[strategy.name] = strategy

    def list_strategies(self) -> List[str]:
        return list(self._strategies.keys())

    def _invoke(self, strategy, llm) -> List[str]:
        """Invoke a strategy's run() method. Returns list of new item IDs."""
        return strategy.run(self._manager, llm=llm)

    def run_all(self, llm=None) -> Dict[str, List[str]]:
        results = {}
        for name, strategy in self._strategies.items():
            new_ids = self._invoke(strategy, llm)
            results[name] = new_ids
            if new_ids:
                if self._recorder:
                    for nid in new_ids:
                        self._recorder.record_consolidation(
                            strategy=name, source_ids=[], result_id=nid,
                        )
                if self._on_consolidation:
                    try:
                        self._on_consolidation(name, new_ids, [])
                    except Exception:
                        pass
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
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _has_new_events(self) -> bool:
        for dq in self._event_queues:
            if dq:
                events.drain(dq)
                return True
        return False

    def _run_loop(self, llm, tick_interval: float):
        last_run: Dict[str, float] = {}
        while self._running:
            self._wake.wait(timeout=tick_interval)
            self._wake.clear()

            has_new = self._has_new_events()
            now = time.time()

            if has_new and self._obs_runner is not None:
                try:
                    self._obs_runner.flush()
                except Exception as e:
                    logger.error("Observation flush failed: %s", e)

            for name, strategy in self._strategies.items():
                last = last_run.get(name, 0)
                elapsed = now - last
                if elapsed >= strategy.interval_seconds or (has_new and elapsed >= 1.0):
                    try:
                        new_ids = self._invoke(strategy, llm)
                        if new_ids:
                            if self._recorder:
                                for nid in new_ids:
                                    self._recorder.record_consolidation(
                                        strategy=name, source_ids=[], result_id=nid,
                                    )
                            if self._on_consolidation:
                                try:
                                    self._on_consolidation(name, new_ids, [])
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.error("Strategy '%s' failed: %s", name, e)
                    last_run[name] = now

    def save(self):
        pass  # No state to persist now that ProcessedIndex is gone
