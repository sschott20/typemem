"""Latency benchmark: measures injection and observation latency across store sizes."""
from __future__ import annotations

import statistics
import tempfile
import time
from dataclasses import dataclass, field

from typemem.injector import MemoryInjector, StageConfig
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-1) of a list of values."""
    if not values:
        return 0.0
    if p <= 0.5:
        return statistics.median(values)
    sorted_vals = sorted(values)
    idx = min(int(len(sorted_vals) * p), len(sorted_vals) - 1)
    return sorted_vals[idx]


@dataclass
class LatencyResult:
    """Latency measurements for a strategy at a given store size."""
    strategy_name: str
    store_size: int
    observation_latencies_ms: list[float] = field(default_factory=list)
    injection_latencies_ms: list[float] = field(default_factory=list)

    @property
    def obs_p50(self) -> float:
        return _percentile(self.observation_latencies_ms, 0.5)

    @property
    def obs_p99(self) -> float:
        return _percentile(self.observation_latencies_ms, 0.99)

    @property
    def inj_p50(self) -> float:
        return _percentile(self.injection_latencies_ms, 0.5)

    @property
    def inj_p99(self) -> float:
        return _percentile(self.injection_latencies_ms, 0.99)


_SAMPLE_OBSERVATIONS = [
    "Red cup on kitchen counter near sink",
    "Blue plate on dining table",
    "Person entered kitchen from hallway",
    "Cup moved to dining table by person",
    "Person left kitchen",
    "Cat jumped on counter near sink",
    "Green bowl in the cabinet above stove",
    "Water bottle next to the refrigerator",
    "Knife on the cutting board",
    "Towel hanging on the oven handle",
    "Dog bowl on the floor near back door",
    "Spoon in the sink",
    "Salt shaker on the counter",
    "Fruit basket on island",
    "Coffee mug on the desk in living room",
    "Book on the shelf",
    "Remote control on the sofa",
    "Keys on the hook by the front door",
    "Umbrella in the stand",
    "Jacket draped over the chair",
]

_SAMPLE_QUERIES = [
    "Where is the cup?",
    "Has anyone been in the kitchen recently?",
    "What objects are on the counter?",
    "Where is the plate?",
    "Is there a cat in the kitchen?",
    "What is on the dining table?",
    "Where are the keys?",
    "What happened in the last few minutes?",
    "Where is the water bottle?",
    "Is anyone in the kitchen?",
]


@dataclass
class _LatencyStrategy:
    name: str
    inject_tiers: list[MemoryTier]
    inject_recency_weight: float
    inject_n_results: int
    dump_all: bool = False


def _seed_store(manager: MemoryManager, start: int, end: int, batch_size: int = 200) -> None:
    """Add memories [start, end) to the store using batch inserts."""
    for b in range(start, end, batch_size):
        b_end = min(b + batch_size, end)
        items = [
            MemoryItem(
                document=f"[t={i}] {_SAMPLE_OBSERVATIONS[i % len(_SAMPLE_OBSERVATIONS)]} (observation #{i})",
                tier=MemoryTier.M1,
                memory_type=MemoryType.OBSERVATION,
                robot_id="bench",
            )
            for i in range(b, b_end)
        ]
        manager.add_batch(items, auto_link=False)


def run_latency_benchmark(
    sizes: list[int] | None = None,
    n_queries: int = 20,
) -> list[LatencyResult]:
    """Measure latency for each strategy across different store sizes."""
    if sizes is None:
        sizes = [100, 500, 1000, 5000, 10000]

    strategies = [
        _LatencyStrategy("full_context", [MemoryTier.M1], 0.0, 100, dump_all=True),
        _LatencyStrategy("monolithic_rag", [MemoryTier.M1], 0.0, 10),
        _LatencyStrategy("tiered_memory", [MemoryTier.M1, MemoryTier.M2], 0.3, 10),
    ]

    all_results: list[LatencyResult] = []

    for strat in strategies:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(persist_dir=tmpdir, robot_id="bench")
            injector = MemoryInjector(manager, cache_ttl=0)
            injector.set_stage_config("bench", StageConfig(
                tiers=strat.inject_tiers,
                max_tokens=400,
                n_results=strat.inject_n_results,
                recency_weight=strat.inject_recency_weight,
            ))

            prev_size = 0
            for size in sorted(sizes):
                _seed_store(manager, prev_size, size)
                prev_size = size

                result = LatencyResult(
                    strategy_name=strat.name,
                    store_size=size,
                )

                # Measure observation latencies
                for i in range(n_queries):
                    obs_text = _SAMPLE_OBSERVATIONS[i % len(_SAMPLE_OBSERVATIONS)]
                    item = MemoryItem(
                        document=f"[bench] {obs_text} (latency test #{i})",
                        tier=MemoryTier.M1,
                        memory_type=MemoryType.OBSERVATION,
                        robot_id="bench",
                    )
                    t0 = time.perf_counter()
                    manager.add(item, auto_link=False)
                    t1 = time.perf_counter()
                    result.observation_latencies_ms.append((t1 - t0) * 1000.0)

                # Measure injection latencies
                for i in range(n_queries):
                    query = _SAMPLE_QUERIES[i % len(_SAMPLE_QUERIES)]
                    t0 = time.perf_counter()
                    if strat.dump_all:
                        items = manager.get_by_tier(MemoryTier.M1)
                        "\n".join(f"[{it.tier}] {it.document}" for it in items)
                    else:
                        injector.inject("bench", query)
                    t1 = time.perf_counter()
                    result.injection_latencies_ms.append((t1 - t0) * 1000.0)

                all_results.append(result)

    return all_results


def latency_results_to_json(results: list[LatencyResult]) -> list[dict]:
    """Convert latency results to JSON-serializable dicts."""
    return [
        {
            "strategy": r.strategy_name,
            "store_size": r.store_size,
            "obs_p50": round(r.obs_p50, 2),
            "obs_p99": round(r.obs_p99, 2),
            "inj_p50": round(r.inj_p50, 2),
            "inj_p99": round(r.inj_p99, 2),
            "observation_latencies_ms": [round(v, 2) for v in r.observation_latencies_ms],
            "injection_latencies_ms": [round(v, 2) for v in r.injection_latencies_ms],
        }
        for r in results
    ]


def print_latency_results(results: list[LatencyResult]) -> None:
    """Pretty-print a latency comparison table."""
    header = (
        f"{'Strategy':<20} {'Size':>6} "
        f"{'Obs p50(ms)':>12} {'Obs p99(ms)':>12} "
        f"{'Inj p50(ms)':>12} {'Inj p99(ms)':>12}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        print(
            f"{r.strategy_name:<20} {r.store_size:>6} "
            f"{r.obs_p50:>12.2f} {r.obs_p99:>12.2f} "
            f"{r.inj_p50:>12.2f} {r.inj_p99:>12.2f}"
        )
    print("=" * len(header))


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    results = run_latency_benchmark(sizes=[100, 500, 1000], n_queries=10)
    print_latency_results(results)

    if "--json" in sys.argv:
        out_path = Path("latency_results.json")
        with open(out_path, "w") as f:
            json.dump(latency_results_to_json(results), f, indent=2)
        print(f"\nResults saved to {out_path}")
