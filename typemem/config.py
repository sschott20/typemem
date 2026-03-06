from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from typemem.store import MemoryStore
from typemem.system import MemorySystem


def load_config(source: Union[str, Path, dict]) -> dict:
    """Load a typemem config from a YAML file path or dict."""
    if isinstance(source, dict):
        return source
    path = Path(source)
    with open(path) as f:
        return yaml.safe_load(f)


def _compile_fn(code: str, expected_name: str = "fn"):
    """Compile a function definition string and return the function object."""
    namespace = {}
    exec(code, namespace)
    if expected_name not in namespace:
        raise ValueError(f"Code block must define a function named '{expected_name}'")
    return namespace[expected_name]


def system_from_config(source: Union[str, Path, dict], store: MemoryStore) -> MemorySystem:
    """Build a MemorySystem from a config file or dict."""
    config = load_config(source)
    system = MemorySystem(store)

    for obs in config.get("observation", []):
        fn = _compile_fn(obs["code"])
        system.add_observation(obs["name"], fn, obs.get("interval", 1.0))

    for cons in config.get("consolidation", []):
        fn = _compile_fn(cons["code"])
        system.add_consolidation(cons["name"], fn, cons.get("interval", 30.0))

    for inj in config.get("injection", []):
        fn = _compile_fn(inj["code"])
        system.add_injection(inj["name"], fn)

    return system
