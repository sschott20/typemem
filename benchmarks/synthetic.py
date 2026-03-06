"""Synthetic benchmark runner for comparing memory strategies."""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from typemem.baselines import make_full_context, make_monolithic_rag
from typemem.chromadb_store import ChromaDBStore
from typemem.system import MemorySystem


@dataclass
class QueryResult:
    """Result of a single query evaluation."""
    time: int
    query: str
    ground_truth: list[str]
    context: str
    token_count: int
    injection_latency_ms: float
    hits: int
    precision: float


@dataclass
class BenchmarkResult:
    """Aggregate result for one strategy on one scenario."""
    strategy_name: str
    scenario_name: str
    query_results: list[QueryResult]
    total_memories: int
    avg_precision: float
    avg_injection_latency_ms: float
    avg_token_count: float


def load_scenario(path: str | Path) -> dict[str, Any]:
    """Load a YAML scenario file."""
    with open(path) as f:
        return yaml.safe_load(f)


def _count_hits(context: str, ground_truth: list[str]) -> int:
    """Count how many ground_truth strings appear in context (case-insensitive)."""
    context_lower = context.lower()
    return sum(1 for gt in ground_truth if gt.lower() in context_lower)


def run_benchmark(
    scenario_path: str | Path,
    system: MemorySystem,
    store: ChromaDBStore,
    strategy_name: str,
    injection_name: str,
    consolidate_interval: int = 10,
) -> BenchmarkResult:
    """Run a full benchmark for a single strategy on a scenario.

    Feeds events in time order, runs consolidation periodically,
    and measures injection latency at query points.
    """
    scenario = load_scenario(scenario_path)
    events = sorted(scenario["events"], key=lambda e: e["time"])
    queries = sorted(scenario["queries"], key=lambda q: q["time"])

    # Merge events and queries into a single timeline
    timeline: list[dict[str, Any]] = []
    for e in events:
        timeline.append({"type": "event", "time": e["time"], "data": e["data"]})
    for q in queries:
        timeline.append({
            "type": "query",
            "time": q["time"],
            "query": q["query"],
            "ground_truth": q["ground_truth"],
            "token_budget": q["token_budget"],
        })
    timeline.sort(key=lambda x: (x["time"], 0 if x["type"] == "event" else 1))

    query_results: list[QueryResult] = []
    events_fed = 0

    for item in timeline:
        if item["type"] == "event":
            system.observe(item["data"])
            events_fed += 1
            # Run consolidation periodically
            if events_fed % consolidate_interval == 0:
                system.consolidate()
        else:
            # Run consolidation before query
            system.consolidate()

            query = item["query"]
            ground_truth = item["ground_truth"]
            token_budget = item["token_budget"]

            # Measure injection latency
            t0 = time.perf_counter()
            context = system.inject(injection_name, query, token_budget)
            t1 = time.perf_counter()
            injection_latency_ms = (t1 - t0) * 1000.0

            token_count = len(context) // 4
            hits = _count_hits(context, ground_truth)
            precision = hits / len(ground_truth) if ground_truth else 0.0

            query_results.append(QueryResult(
                time=item["time"],
                query=query,
                ground_truth=ground_truth,
                context=context,
                token_count=token_count,
                injection_latency_ms=injection_latency_ms,
                hits=hits,
                precision=precision,
            ))

    total_memories = store.count()
    avg_precision = (
        sum(qr.precision for qr in query_results) / len(query_results)
        if query_results else 0.0
    )
    avg_injection_latency_ms = (
        sum(qr.injection_latency_ms for qr in query_results) / len(query_results)
        if query_results else 0.0
    )
    avg_token_count = (
        sum(qr.token_count for qr in query_results) / len(query_results)
        if query_results else 0.0
    )

    return BenchmarkResult(
        strategy_name=strategy_name,
        scenario_name=scenario["name"],
        query_results=query_results,
        total_memories=total_memories,
        avg_precision=avg_precision,
        avg_injection_latency_ms=avg_injection_latency_ms,
        avg_token_count=avg_token_count,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    """Pretty-print a comparison table of benchmark results."""
    header = f"{'Strategy':<20} {'Scenario':<35} {'Memories':>8} {'AvgPrec':>8} {'AvgLat(ms)':>11} {'AvgTokens':>10}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        print(
            f"{r.strategy_name:<20} {r.scenario_name:<35} "
            f"{r.total_memories:>8} {r.avg_precision:>8.2f} "
            f"{r.avg_injection_latency_ms:>11.2f} {r.avg_token_count:>10.1f}"
        )
    print("=" * len(header))

    # Per-query detail
    for r in results:
        print(f"\n--- {r.strategy_name} ---")
        for qr in r.query_results:
            print(
                f"  t={qr.time:>4}s  query={qr.query!r:<50} "
                f"hits={qr.hits}/{len(qr.ground_truth)}  "
                f"prec={qr.precision:.2f}  "
                f"lat={qr.injection_latency_ms:.2f}ms  "
                f"tokens={qr.token_count}"
            )


if __name__ == "__main__":
    scenario_path = Path(__file__).parent / "scenarios" / "kitchen_patrol.yaml"

    strategies = [
        ("full_context", make_full_context, "dump"),
        ("monolithic_rag", make_monolithic_rag, "topk"),
    ]

    results: list[BenchmarkResult] = []
    for strategy_name, factory, injection_name in strategies:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChromaDBStore(persist_dir=tmpdir)
            system = factory(store)
            result = run_benchmark(
                scenario_path=scenario_path,
                system=system,
                store=store,
                strategy_name=strategy_name,
                injection_name=injection_name,
            )
            results.append(result)

    print_results(results)
