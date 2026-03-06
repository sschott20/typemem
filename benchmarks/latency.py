"""Latency benchmark: measures injection and observation latency across store sizes."""
from __future__ import annotations

import statistics
import tempfile
import time
from dataclasses import dataclass, field

from typemem.baselines import make_full_context, make_monolithic_rag
from typemem.chromadb_store import ChromaDBStore
from typemem.system import MemorySystem


@dataclass
class LatencyResult:
    """Latency measurements for a strategy at a given store size."""
    strategy_name: str
    store_size: int
    observation_latencies_ms: list[float] = field(default_factory=list)
    injection_latencies_ms: list[float] = field(default_factory=list)

    @property
    def obs_p50(self) -> float:
        if not self.observation_latencies_ms:
            return 0.0
        return statistics.median(self.observation_latencies_ms)

    @property
    def obs_p99(self) -> float:
        if not self.observation_latencies_ms:
            return 0.0
        sorted_vals = sorted(self.observation_latencies_ms)
        idx = int(len(sorted_vals) * 0.99)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    @property
    def inj_p50(self) -> float:
        if not self.injection_latencies_ms:
            return 0.0
        return statistics.median(self.injection_latencies_ms)

    @property
    def inj_p99(self) -> float:
        if not self.injection_latencies_ms:
            return 0.0
        sorted_vals = sorted(self.injection_latencies_ms)
        idx = int(len(sorted_vals) * 0.99)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]


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


def _seed_store(store: ChromaDBStore, n: int, batch_size: int = 100) -> None:
    """Add n diverse memories to the store using batched inserts for speed."""
    from typemem.types import make_id

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ids = [make_id() for _ in range(end - start)]
        docs = [
            f"[t={i}] {_SAMPLE_OBSERVATIONS[i % len(_SAMPLE_OBSERVATIONS)]} (observation #{i})"
            for i in range(start, end)
        ]
        # Use ChromaDB collection's batch add directly for efficiency
        store._collection.add(ids=ids, documents=docs)


def run_latency_benchmark(
    sizes: list[int] | None = None,
    n_queries: int = 20,
) -> list[LatencyResult]:
    """Measure latency for each strategy across different store sizes."""
    if sizes is None:
        sizes = [100, 500, 1000, 5000, 10000]

    strategies = [
        ("full_context", make_full_context, "dump"),
        ("monolithic_rag", make_monolithic_rag, "topk"),
    ]

    all_results: list[LatencyResult] = []

    for strategy_name, factory, injection_name in strategies:
        for size in sizes:
            with tempfile.TemporaryDirectory() as tmpdir:
                store = ChromaDBStore(persist_dir=tmpdir)
                system = factory(store)

                # Seed the store
                _seed_store(store, size)

                result = LatencyResult(
                    strategy_name=strategy_name,
                    store_size=size,
                )

                # Measure observation latencies
                for i in range(n_queries):
                    obs = _SAMPLE_OBSERVATIONS[i % len(_SAMPLE_OBSERVATIONS)]
                    data = {"text": f"[bench] {obs} (latency test #{i})"}
                    t0 = time.perf_counter()
                    system.observe(data)
                    t1 = time.perf_counter()
                    result.observation_latencies_ms.append((t1 - t0) * 1000.0)

                # Measure injection latencies
                for i in range(n_queries):
                    query = _SAMPLE_QUERIES[i % len(_SAMPLE_QUERIES)]
                    t0 = time.perf_counter()
                    system.inject(injection_name, query, token_budget=400)
                    t1 = time.perf_counter()
                    result.injection_latencies_ms.append((t1 - t0) * 1000.0)

                all_results.append(result)

    return all_results


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
    results = run_latency_benchmark(sizes=[100, 500, 1000], n_queries=10)
    print_latency_results(results)
