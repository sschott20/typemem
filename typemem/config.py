"""typemem configuration and YAML config loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

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


def parse_stage_configs(stages_raw: dict) -> Dict[str, StageConfig]:
    """Parse a stages dict (from YAML) into StageConfig objects.

    Expected format per stage::

        S1:
          tiers: [M1, M2]
          max_tokens: 400
          pinned_sources: [lesson_s1]
          live_sources: [observation]
    """
    from typemem.memory_item import MemoryTier
    tier_map = {t.label: t for t in MemoryTier}

    stage_configs: Dict[str, StageConfig] = {}
    for stage_name, stage_data in stages_raw.items():
        raw_tiers = stage_data.get("tiers", [])
        tiers = []
        for t in raw_tiers:
            if t not in tier_map:
                raise ValueError(
                    f"Unknown tier '{t}' in stage '{stage_name}'. "
                    f"Valid tiers: {list(tier_map)}"
                )
            tiers.append(tier_map[t])
        stage_configs[stage_name] = StageConfig(
            tiers=tiers,
            max_tokens=stage_data.get("max_tokens", 500),
            n_results=stage_data.get("n_results", 10),
            recency_weight=stage_data.get("recency_weight", 0.3),
            pinned_sources=stage_data.get("pinned_sources", []),
            live_sources=stage_data.get("live_sources", []),
        )
    return stage_configs


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

    stages_raw = mem_cfg.get("stages", {})
    stage_configs = parse_stage_configs(stages_raw) if stages_raw else None

    config = MemoryConfig(
        link_top_k=mem_cfg.get("link_top_k", LINK_TOP_K),
        link_distance_threshold=mem_cfg.get("link_distance_threshold", LINK_DISTANCE_THRESHOLD),
        dedup_distance_threshold=mem_cfg.get("dedup_distance_threshold", DEDUP_DISTANCE_THRESHOLD),
        expiry_interval=mem_cfg.get("expiry_interval", 60.0),
        injector_cache_ttl=mem_cfg.get("injector_cache_ttl", _DEFAULT_CACHE_TTL),
        stage_configs=stage_configs,
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
