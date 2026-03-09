"""Tracing event dataclasses for memory visualization."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StoreEvent:
    """A single store operation."""

    operation: str
    details: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConsolidationEvent:
    """A consolidation run with inputs/outputs."""

    name: str
    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    deletions: list[str]
    duration_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class InjectionEvent:
    """An injection query with results."""

    name: str
    query: str
    token_budget: int
    search_results: list[dict[str, Any]]
    context: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)
