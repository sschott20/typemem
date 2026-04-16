"""typemem — Abstract memory framework for LLM-controlled robots."""

import os
from typing import List, Optional, Union

from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.link_index import LinkIndex
from typemem.consolidation import ConsolidationEngine
from typemem.injector import MemoryInjector, StageConfig
from typemem.recorder import SessionRecorder
from typemem.frame_store import FrameStore
from typemem.plugins.base import ObservationPlugin, ConsolidationPlugin, InjectionPlugin
from typemem.plugins.base_injection import BaseInjectionPlugin, InjectionSpec
from typemem.plugins.runner import ObservationRunner
from typemem.plugins.loader import PluginLoader
from typemem.plugins.text_summary import TextSummaryPlugin
from typemem.plugins.llm_summary import LLMSummaryPlugin
from typemem.plugins.tier_retention_gc import TierRetentionGC
from typemem.llm import LLMCallable, make_anthropic_llm
from typemem.config import MemoryConfig, system_from_config
from typemem import events


def create_memory_system(
    persist_dir: str,
    robot_id: str,
    plugins: Optional[List[Union[ObservationPlugin, ConsolidationPlugin]]] = None,
    recording_path: Optional[str] = None,
    frame_store_dir: Optional[str] = None,
    config: Optional[MemoryConfig] = None,
):
    """Create a complete memory system with all components wired together.

    Args:
        plugins: Explicit list of observation and consolidation plugins.
                 None means no plugins are loaded (no auto-discovery).

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

    # Sort plugins by type
    obs_runner = ObservationRunner()
    engine = ConsolidationEngine(manager)

    # Collect injection plugins separately — registered after injector is created
    injection_plugins: List[InjectionPlugin] = []
    for p in (plugins or []):
        if isinstance(p, ObservationPlugin):
            obs_runner.register(p)
        elif isinstance(p, ConsolidationPlugin):
            engine.register_strategy(p)
        elif isinstance(p, InjectionPlugin):
            injection_plugins.append(p)

    engine.set_observation_runner(obs_runner)

    injector = MemoryInjector(manager, cache_ttl=cfg.injector_cache_ttl)
    injector.set_runner(obs_runner)

    for ip in injection_plugins:
        injector.register_plugin(ip)

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
