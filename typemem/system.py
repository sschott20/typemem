from __future__ import annotations

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

from typemem.store import MemoryStore
from typemem.types import ObservationFn, ConsolidationFn, InjectionFn


class MemorySystem:
    """Registry and runner for observation, consolidation, and injection functions."""

    def __init__(self, store: MemoryStore):
        self.store = store
        self._observations: dict[str, tuple[ObservationFn, float]] = {}
        self._consolidations: dict[str, tuple[ConsolidationFn, float]] = {}
        self._injections: dict[str, InjectionFn] = {}
        self._running = False
        self._threads: list[threading.Thread] = []

    def add_observation(self, name: str, fn: ObservationFn, interval: float) -> None:
        self._observations[name] = (fn, interval)

    def add_consolidation(self, name: str, fn: ConsolidationFn, interval: float) -> None:
        self._consolidations[name] = (fn, interval)

    def add_injection(self, name: str, fn: InjectionFn) -> None:
        self._injections[name] = fn

    def observe(self, raw_data: dict) -> list[str]:
        """Run all observation functions with given raw data. Returns all created IDs."""
        all_ids = []
        for fn, _ in self._observations.values():
            ids = fn(raw_data, self.store)
            all_ids.extend(ids)
        return all_ids

    def consolidate(self) -> list[str]:
        """Run all consolidation functions. Returns all created IDs."""
        all_ids = []
        for fn, _ in self._consolidations.values():
            ids = fn(self.store)
            all_ids.extend(ids)
        return all_ids

    def inject(self, name: str, query: str, token_budget: int) -> str:
        """Run a named injection function."""
        if name not in self._injections:
            raise KeyError(f"No injection function named '{name}'")
        return self._injections[name](query, self.store, token_budget)

    def start(self, data_source: Callable[[], dict]) -> None:
        """Start background threads for observation and consolidation.

        Not thread-safe: call from a single thread only. The spawned daemon
        threads share ``self.store``, so the underlying store implementation
        must be safe for concurrent reads/writes.
        """
        self._running = True
        obs_thread = threading.Thread(
            target=self._obs_loop, args=(data_source,), daemon=True,
        )
        cons_thread = threading.Thread(
            target=self._cons_loop, daemon=True,
        )
        obs_thread.start()
        cons_thread.start()
        self._threads = [obs_thread, cons_thread]

    def stop(self) -> None:
        """Stop background threads."""
        self._running = False
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads = []

    def _obs_loop(self, data_source: Callable[[], dict]) -> None:
        last_run: dict[str, float] = {}
        while self._running:
            now = time.time()
            for name, (fn, interval) in self._observations.items():
                if now - last_run.get(name, 0) >= interval:
                    try:
                        raw_data = data_source()
                        fn(raw_data, self.store)
                    except Exception:
                        logger.warning("Observation '%s' failed", name, exc_info=True)
                    last_run[name] = now
            time.sleep(0.1)

    def _cons_loop(self) -> None:
        last_run: dict[str, float] = {}
        while self._running:
            now = time.time()
            for name, (fn, interval) in self._consolidations.items():
                if now - last_run.get(name, 0) >= interval:
                    try:
                        fn(self.store)
                    except Exception:
                        logger.warning("Consolidation '%s' failed", name, exc_info=True)
                    last_run[name] = now
            time.sleep(0.5)
