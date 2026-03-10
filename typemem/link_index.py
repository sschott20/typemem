import json
import logging
import os
from collections import defaultdict
from typing import Dict, Set

logger = logging.getLogger(__name__)


class LinkIndex:
    """Zettelkasten-style bidirectional link index between memory items.
    Persisted as JSON. Links are always bidirectional.
    """

    def __init__(self, path: str):
        self._path = path
        self._links: Dict[str, Set[str]] = defaultdict(set)
        if os.path.exists(path):
            self._load()

    def add_link(self, id_a: str, id_b: str):
        if id_a == id_b:
            return
        self._links[id_a].add(id_b)
        self._links[id_b].add(id_a)

    def remove_link(self, id_a: str, id_b: str):
        self._links[id_a].discard(id_b)
        self._links[id_b].discard(id_a)

    def remove_node(self, node_id: str):
        neighbors = set(self._links.get(node_id, set()))
        for neighbor in neighbors:
            self._links[neighbor].discard(node_id)
        self._links.pop(node_id, None)

    def get_links(self, node_id: str) -> Set[str]:
        return set(self._links.get(node_id, set()))

    def link_count(self, node_id: str) -> int:
        return len(self._links.get(node_id, set()))

    def save(self):
        serializable = {k: list(v) for k, v in self._links.items() if v}
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(serializable, f)

    def _load(self):
        try:
            with open(self._path, "r") as f:
                data = json.load(f)
            self._links = defaultdict(set, {k: set(v) for k, v in data.items()})
        except Exception as e:
            logger.error("LinkIndex failed to load %s: %s", self._path, e)
            self._links = defaultdict(set)
