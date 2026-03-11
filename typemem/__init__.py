"""typemem — Abstract memory framework for LLM-controlled robots."""

import os
from typing import List, Optional, Union

from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.link_index import LinkIndex
from typemem.processed_index import ProcessedIndex
from typemem.consolidation import ConsolidationEngine
from typemem.injector import MemoryInjector, StageConfig
from typemem.recorder import SessionRecorder
from typemem.frame_store import FrameStore
from typemem.plugins.base import ObservationPlugin, ConsolidationPlugin
from typemem.plugins.runner import ObservationRunner
from typemem.plugins.loader import PluginLoader
from typemem.plugins.text_summary import TextSummaryPlugin
from typemem.plugins.llm_summary import LLMSummaryPlugin
from typemem.llm import LLMCallable, make_anthropic_llm
from typemem.config import MemoryConfig, system_from_config
from typemem import events


def create_memory_system(
    persist_dir: str,
    robot_id: str,
    recording_path: Optional[str] = None,
    frame_store_dir: Optional[str] = None,
    disabled_plugins: Optional[List[str]] = None,
    extra_plugins: Optional[List[Union[ObservationPlugin, ConsolidationPlugin]]] = None,
    obs_package: Optional[str] = None,
    consol_package: Optional[str] = None,
    config: Optional[MemoryConfig] = None,
):
    """Create a complete memory system with all components wired together.

    Returns:
        Tuple of (MemoryManager, ConsolidationEngine, MemoryInjector,
                  SessionRecorder, ObservationRunner, FrameStore | None)
    """
    cfg = config or MemoryConfig()

    manager = MemoryManager(
        persist_dir=persist_dir,
        robot_id=robot_id,
        link_top_k=cfg.link_top_k,
        link_distance_threshold=cfg.link_distance_threshold,
        dedup_distance_threshold=cfg.dedup_distance_threshold,
    )

    # Auto-discover plugins from specified packages
    obs_plugins, consol_plugins = PluginLoader.discover(
        obs_package=obs_package,
        consol_package=consol_package,
        disabled=disabled_plugins or [],
    )

    # Add any extra plugins
    for p in (extra_plugins or []):
        if isinstance(p, ObservationPlugin):
            obs_plugins.append(p)
        elif isinstance(p, ConsolidationPlugin):
            consol_plugins.append(p)

    # Wire up observation runner
    obs_runner = ObservationRunner()
    for p in obs_plugins:
        obs_runner.register(p)

    # Wire up consolidation engine
    engine = ConsolidationEngine(manager, expiry_interval=cfg.expiry_interval)
    for p in consol_plugins:
        engine.register_strategy(p)

    injector = MemoryInjector(manager, cache_ttl=cfg.injector_cache_ttl)

    if cfg.stage_configs:
        for stage, stage_cfg in cfg.stage_configs.items():
            injector.set_stage_config(stage, stage_cfg)

    recorder = SessionRecorder(
        path=recording_path or os.devnull,
        enabled=recording_path is not None,
    )

    manager.set_recorder(recorder)
    engine.set_recorder(recorder)
    injector.set_recorder(recorder)

    frame_store = None
    if frame_store_dir:
        frame_store = FrameStore(frame_store_dir)

    return manager, engine, injector, recorder, obs_runner, frame_store
