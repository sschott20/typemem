"""Per-plugin processed ID tracking with JSON persistence."""

import json
import os
from typing import Dict, List, Set


class ProcessedIndex:
    """Tracks which memory item IDs each consolidation plugin has processed.
    Stores {plugin_name: set_of_ids} in memory. Flushed to disk via save().
    """

    def __init__(self, path: str):
        self._path = path
        self._data: Dict[str, Set[str]] = {}
        if os.path.exists(path):
            with open(path) as f:
                raw = json.load(f)
            self._data = {k: set(v) for k, v in raw.items()}

    def is_processed(self, plugin_name: str, item_id: str) -> bool:
        return item_id in self._data.get(plugin_name, set())

    def mark_processed(self, plugin_name: str, item_ids: List[str]) -> None:
        if plugin_name not in self._data:
            self._data[plugin_name] = set()
        self._data[plugin_name].update(item_ids)

    def filter_unprocessed(self, plugin_name: str, item_ids: List[str]) -> List[str]:
        processed = self._data.get(plugin_name, set())
        return [i for i in item_ids if i not in processed]

    def prune(self, live_ids: Set[str]) -> None:
        for plugin_name in self._data:
            self._data[plugin_name] &= live_ids

    def save(self) -> None:
        with open(self._path, "w") as f:
            json.dump({k: sorted(v) for k, v in self._data.items()}, f)
