import logging
import time
import threading
from typing import Dict, List, Optional

from typemem.plugins.base import ObservationPlugin
from typemem.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class ObservationRunner:
    """Schedules and runs observation plugins in a daemon thread."""

    def __init__(self):
        self._plugins: Dict[str, ObservationPlugin] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def register(self, plugin: ObservationPlugin) -> None:
        self._plugins[plugin.name] = plugin

    def list_plugins(self) -> List[str]:
        return list(self._plugins.keys())

    def start(
        self,
        memory_manager: MemoryManager,
        robot_id: str,
        tick_interval: float = 0.1,
    ) -> None:
        for plugin in self._plugins.values():
            plugin.setup(memory_manager, robot_id)

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, args=(tick_interval,), daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

        for plugin in self._plugins.values():
            try:
                plugin.teardown()
            except Exception as e:
                logger.error("Plugin '%s' teardown error: %s", plugin.name, e)

    def flush(self) -> None:
        """Run all plugins once (synchronous). Used by consolidation engine
        to ensure M1 is up-to-date before reading."""
        for name, plugin in self._plugins.items():
            try:
                plugin.run()
            except Exception as e:
                logger.error("Plugin '%s' flush error: %s", name, e)

    def get_live_summary(self, name: str) -> Optional[str]:
        """Get live summary from a plugin by name. Returns None if plugin not found or has no data."""
        plugin = self._plugins.get(name)
        if plugin is None:
            return None
        try:
            return plugin.live_summary()
        except Exception:
            logger.error("Plugin '%s' live_summary error", name, exc_info=True)
            return None

    def get_all_live_summaries(self) -> Dict[str, str]:
        """Get all available live summaries. Used by viz server."""
        results = {}
        for name, plugin in self._plugins.items():
            try:
                value = plugin.live_summary()
                if value:
                    results[name] = value
            except Exception:
                logger.debug("Plugin '%s' live_summary raised, skipping", name)
        return results

    def _run_loop(self, tick_interval: float) -> None:
        last_run: Dict[str, float] = {}
        while self._running:
            now = time.time()
            for name, plugin in self._plugins.items():
                last = last_run.get(name, 0.0)
                if now - last >= plugin.interval_seconds:
                    try:
                        plugin.run()
                    except Exception as e:
                        logger.error("Plugin '%s' run error: %s", name, e)
                    last_run[name] = now
            time.sleep(tick_interval)
