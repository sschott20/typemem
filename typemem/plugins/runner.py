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
