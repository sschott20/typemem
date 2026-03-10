"""typemem configuration and YAML config loader."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from typemem.injector import StageConfig, _DEFAULT_CACHE_TTL
from typemem.memory_manager import LINK_TOP_K, LINK_DISTANCE_THRESHOLD, DEDUP_DISTANCE_THRESHOLD


@dataclass
class MemoryConfig:
    """Centralized configuration for the memory system."""
    link_top_k: int = LINK_TOP_K
    link_distance_threshold: float = LINK_DISTANCE_THRESHOLD
    dedup_distance_threshold: float = DEDUP_DISTANCE_THRESHOLD
    expiry_interval: float = 60.0
    stage_configs: Optional[Dict[str, StageConfig]] = None
    injector_cache_ttl: float = _DEFAULT_CACHE_TTL


def _load_yaml(source: Union[str, Path, dict]) -> dict:
    if isinstance(source, dict):
        return source
    path = Path(source)
    with open(path) as f:
        return yaml.safe_load(f) or {}


def system_from_config(
    source: Union[str, Path, dict],
    robot_id: str,
    persist_dir: str,
    obs_package: Optional[str] = None,
    consol_package: Optional[str] = None,
):
    """Build a complete memory system from a config file or dict.

    Returns: (MemoryManager, ConsolidationEngine, MemoryInjector,
              SessionRecorder, ObservationRunner, FrameStore | None)
    """
    from typemem import create_memory_system

    raw = _load_yaml(source)
    mem_cfg = raw.get("memory", {})

    config = MemoryConfig(
        link_top_k=mem_cfg.get("link_top_k", LINK_TOP_K),
        link_distance_threshold=mem_cfg.get("link_distance_threshold", LINK_DISTANCE_THRESHOLD),
        dedup_distance_threshold=mem_cfg.get("dedup_distance_threshold", DEDUP_DISTANCE_THRESHOLD),
        expiry_interval=mem_cfg.get("expiry_interval", 60.0),
        injector_cache_ttl=mem_cfg.get("injector_cache_ttl", _DEFAULT_CACHE_TTL),
    )

    return create_memory_system(
        persist_dir=persist_dir,
        robot_id=robot_id,
        recording_path=raw.get("recording_path"),
        frame_store_dir=raw.get("frame_store_dir"),
        disabled_plugins=raw.get("disabled_plugins"),
        obs_package=obs_package or raw.get("obs_package"),
        consol_package=consol_package or raw.get("consol_package"),
        config=config,
    )
