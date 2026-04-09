"""Synthetic benchmark runner for comparing memory strategies."""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from typemem.consolidation import ConsolidationEngine
from typemem.injector import MemoryInjector, StageConfig
from typemem.llm import LLMCallable
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.plugins.base import ConsolidationPlugin
from typemem.plugins.llm_summary import LLMSummaryPlugin
from typemem.plugins.text_summary import TextSummaryPlugin

_BENCH_STAGE = "bench"


@dataclass
class StrategyConfig:
    """Configuration for a benchmark strategy."""
    name: str
    obs_tier: MemoryTier
    consol_plugins: list[ConsolidationPlugin]
    inject_tiers: list[MemoryTier]
    inject_recency_weight: float
    inject_n_results: int
    dump_all: bool = False

    def make_fresh_plugins(self) -> list[ConsolidationPlugin]:
        """Return fresh plugin instances (processed state must not leak between runs)."""
        plugins = []
        for p in self.consol_plugins:
            if isinstance(p, TextSummaryPlugin):
                plugins.append(TextSummaryPlugin(
                    batch_size=p._batch_size, interval=p._interval,
                ))
            elif isinstance(p, LLMSummaryPlugin):
                plugins.append(LLMSummaryPlugin(
                    batch_size=p._batch_size, interval=p._interval,
                ))
            else:
                plugins.append(p)
        return plugins


def _default_strategies() -> list[StrategyConfig]:
    return [
        StrategyConfig(
            "full_context", MemoryTier.M1, [],
            [MemoryTier.M1], 0.0, 100, dump_all=True,
        ),
        StrategyConfig(
            "monolithic_rag", MemoryTier.M1, [],
            [MemoryTier.M1], 0.0, 10,
        ),
        StrategyConfig(
            "tiered_memory", MemoryTier.M1,
            [TextSummaryPlugin(batch_size=3, interval=0)],
            [MemoryTier.M1, MemoryTier.M2], 0.3, 10,
        ),
        StrategyConfig(
            "tiered_llm", MemoryTier.M1,
            [LLMSummaryPlugin(batch_size=3, interval=0)],
            [MemoryTier.M1, MemoryTier.M2], 0.3, 10,
        ),
        StrategyConfig(
            "rag_with_recency", MemoryTier.M1, [],
            [MemoryTier.M1], 0.5, 10,
        ),
        StrategyConfig(
            "tiered_no_consol", MemoryTier.M1, [],
            [MemoryTier.M1, MemoryTier.M2], 0.3, 10,
        ),
    ]


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
    llm_judge_score: float | None = None


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
    avg_llm_judge_score: float | None = None


def load_scenario(path: str | Path) -> dict[str, Any]:
    """Load a YAML scenario file."""
    with open(path) as f:
        return yaml.safe_load(f)


def _count_hits(context: str, ground_truth: list[str]) -> int:
    """Count how many ground_truth strings appear in context (case-insensitive)."""
    context_lower = context.lower()
    return sum(1 for gt in ground_truth if gt.lower() in context_lower)


def _dump_all(manager: MemoryManager, tiers: list[MemoryTier], token_budget: int) -> str:
    """Full-context injection: concatenate all memories, truncate to budget."""
    lines = []
    for tier in tiers:
        for item in manager.get_by_tier(tier):
            lines.append(f"[{item.tier}] {item.document}")
    context = "\n".join(lines)
    max_chars = token_budget * 4
    return context[:max_chars]


def run_benchmark(
    scenario_path: str | Path,
    strategy: StrategyConfig,
    consolidate_interval: int = 10,
    llm: LLMCallable | None = None,
) -> BenchmarkResult:
    """Run a full benchmark for a single strategy on a scenario."""
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

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MemoryManager(persist_dir=tmpdir, robot_id="bench")
        engine = ConsolidationEngine(manager, expiry_interval=9999)
        for plugin in strategy.make_fresh_plugins():
            engine.register_strategy(plugin)

        injector = MemoryInjector(manager, cache_ttl=0)
        injector.set_stage_config(_BENCH_STAGE, StageConfig(
            tiers=strategy.inject_tiers,
            max_tokens=1000,  # overridden per-query
            n_results=strategy.inject_n_results,
            recency_weight=strategy.inject_recency_weight,
        ))

        query_results: list[QueryResult] = []
        events_fed = 0

        for item in timeline:
            if item["type"] == "event":
                text = item["data"].get("text", str(item["data"]))
                mem_item = MemoryItem(
                    document=text,
                    tier=strategy.obs_tier,
                    memory_type=MemoryType.OBSERVATION,
                    robot_id="bench",
                )
                manager.add(mem_item)
                events_fed += 1
                if events_fed % consolidate_interval == 0:
                    engine.run_all(llm=llm)
            else:
                engine.run_all(llm=llm)

                query = item["query"]
                ground_truth = item["ground_truth"]
                token_budget = item["token_budget"]

                t0 = time.perf_counter()
                if strategy.dump_all:
                    context = _dump_all(manager, strategy.inject_tiers, token_budget)
                else:
                    context = injector.inject(
                        _BENCH_STAGE, query, max_tokens=token_budget,
                    )
                t1 = time.perf_counter()
                injection_latency_ms = (t1 - t0) * 1000.0

                token_count = len(context) // 4
                hits = _count_hits(context, ground_truth)
                precision = hits / len(ground_truth) if ground_truth else 0.0

                judge_score = None
                if llm is not None:
                    from benchmarks.llm_judge import llm_judge_score
                    judge_score = llm_judge_score(llm, query, context, ground_truth)

                query_results.append(QueryResult(
                    time=item["time"],
                    query=query,
                    ground_truth=ground_truth,
                    context=context,
                    token_count=token_count,
                    injection_latency_ms=injection_latency_ms,
                    hits=hits,
                    precision=precision,
                    llm_judge_score=judge_score,
                ))

        total_memories = manager.count()

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

    judge_scores = [qr.llm_judge_score for qr in query_results if qr.llm_judge_score is not None]
    avg_llm_judge_score = (
        sum(judge_scores) / len(judge_scores) if judge_scores else None
    )

    return BenchmarkResult(
        strategy_name=strategy.name,
        scenario_name=scenario["name"],
        query_results=query_results,
        total_memories=total_memories,
        avg_precision=avg_precision,
        avg_injection_latency_ms=avg_injection_latency_ms,
        avg_token_count=avg_token_count,
        avg_llm_judge_score=avg_llm_judge_score,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    """Pretty-print a comparison table of benchmark results."""
    has_judge = any(r.avg_llm_judge_score is not None for r in results)
    header = f"{'Strategy':<20} {'Scenario':<35} {'Memories':>8} {'AvgPrec':>8} {'AvgLat(ms)':>11} {'AvgTokens':>10}"
    if has_judge:
        header += f" {'AvgJudge':>9}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        line = (
            f"{r.strategy_name:<20} {r.scenario_name:<35} "
            f"{r.total_memories:>8} {r.avg_precision:>8.2f} "
            f"{r.avg_injection_latency_ms:>11.2f} {r.avg_token_count:>10.1f}"
        )
        if has_judge:
            if r.avg_llm_judge_score is not None:
                line += f" {r.avg_llm_judge_score:>9.2f}"
            else:
                line += f" {'N/A':>9}"
        print(line)
    print("=" * len(header))

    # Per-query detail
    for r in results:
        print(f"\n--- {r.strategy_name} ---")
        for qr in r.query_results:
            detail = (
                f"  t={qr.time:>4}s  query={qr.query!r:<50} "
                f"hits={qr.hits}/{len(qr.ground_truth)}  "
                f"prec={qr.precision:.2f}  "
                f"lat={qr.injection_latency_ms:.2f}ms  "
                f"tokens={qr.token_count}"
            )
            if has_judge and qr.llm_judge_score is not None:
                detail += f"  judge={qr.llm_judge_score:.1f}"
            print(detail)


def results_to_json(results: list[BenchmarkResult]) -> list[dict]:
    """Convert results to JSON-serializable dicts."""
    out = []
    for r in results:
        out.append({
            "strategy": r.strategy_name,
            "scenario": r.scenario_name,
            "total_memories": r.total_memories,
            "avg_precision": round(r.avg_precision, 4),
            "avg_injection_latency_ms": round(r.avg_injection_latency_ms, 2),
            "avg_token_count": round(r.avg_token_count, 1),
            "avg_llm_judge_score": r.avg_llm_judge_score,
            "queries": [
                {
                    "time": qr.time,
                    "query": qr.query,
                    "hits": qr.hits,
                    "total_ground_truth": len(qr.ground_truth),
                    "precision": round(qr.precision, 4),
                    "injection_latency_ms": round(qr.injection_latency_ms, 2),
                    "token_count": qr.token_count,
                    "llm_judge_score": qr.llm_judge_score,
                }
                for qr in r.query_results
            ],
        })
    return out


def run_all_scenarios(
    scenario_dir: str | Path | None = None,
    strategies: list[StrategyConfig] | None = None,
    llm: LLMCallable | None = None,
) -> list[BenchmarkResult]:
    """Run all strategies against all scenarios in a directory."""
    if scenario_dir is None:
        scenario_dir = Path(__file__).parent / "scenarios"
    scenario_dir = Path(scenario_dir)
    if strategies is None:
        strategies = _default_strategies()

    scenario_files = sorted(scenario_dir.glob("*.yaml"))
    all_results: list[BenchmarkResult] = []

    for scenario_path in scenario_files:
        for strategy in strategies:
            result = run_benchmark(
                scenario_path=scenario_path,
                strategy=strategy,
                llm=llm,
            )
            all_results.append(result)

    return all_results


if __name__ == "__main__":
    import json
    import sys

    llm = None
    if "--llm" in sys.argv:
        from typemem.llm import make_anthropic_llm
        llm = make_anthropic_llm()
        print("Using LLM judge (Haiku 4.5) for evaluation")

    results = run_all_scenarios(llm=llm)
    print_results(results)

    if "--json" in sys.argv:
        out_path = Path("benchmark_results.json")
        with open(out_path, "w") as f:
            json.dump(results_to_json(results), f, indent=2)
        print(f"\nResults saved to {out_path}")
