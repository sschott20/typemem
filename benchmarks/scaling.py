"""Quality-vs-store-size scaling benchmark.

Two modes:
  - replay_fraction: feed first N% of events, then query
  - noise_padding: feed all events + N distractors, then query
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.synthetic import (
    StrategyConfig,
    _count_hits,
    _dump_all,
    _BENCH_STAGE,
    _default_strategies,
    load_scenario,
)
from typemem.consolidation import ConsolidationEngine
from typemem.injector import MemoryInjector, StageConfig
from typemem.llm import LLMCallable
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager


DISTRACTOR_POOL = [
    "Chair in conference room B on floor 3",
    "Printer on 3rd floor near elevator",
    "Whiteboard marker in meeting room A",
    "Trash can full in break room",
    "Fire extinguisher inspected in hallway",
    "Elevator doors closing on floor 2",
    "Light flickering in parking garage",
    "Water cooler empty in lobby",
    "Cleaning cart in west corridor",
    "Security badge found on floor 1",
    "Vending machine out of order",
    "Window open in office 201",
    "Delivery package at reception desk",
    "Ceiling tile damaged in restroom",
    "Air conditioning vent blocked in room 305",
    "Phone ringing in empty office",
    "Mop and bucket in supply closet",
    "Smoke detector test in east wing",
    "Emergency exit sign dim on floor 4",
    "Coffee spill in hallway near kitchen",
    "Plant wilting in reception area",
    "Filing cabinet left open in office 102",
    "Monitor left on in server room",
    "Stairwell door propped open on floor 2",
    "Bicycle locked in unauthorized area",
]


@dataclass
class ScalingResult:
    strategy_name: str
    scenario_name: str
    mode: str
    scale_param: float
    store_size: int
    avg_precision: float
    avg_llm_judge_score: float | None
    avg_injection_latency_ms: float


def _run_with_events(
    scenario: dict[str, Any],
    events: list[dict],
    strategy: StrategyConfig,
    llm: LLMCallable | None = None,
    consolidate_interval: int = 10,
) -> ScalingResult:
    """Run queries against a store built from the given events."""
    queries = sorted(scenario["queries"], key=lambda q: q["time"])

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MemoryManager(persist_dir=tmpdir, robot_id="bench")
        engine = ConsolidationEngine(manager, expiry_interval=9999)
        for plugin in strategy.make_fresh_plugins():
            engine.register_strategy(plugin)

        injector = MemoryInjector(manager, cache_ttl=0)
        injector.set_stage_config(_BENCH_STAGE, StageConfig(
            tiers=strategy.inject_tiers,
            max_tokens=1000,
            n_results=strategy.inject_n_results,
            recency_weight=strategy.inject_recency_weight,
        ))

        # Feed all events
        for i, event in enumerate(events):
            text = event["data"].get("text", str(event["data"]))
            mem_item = MemoryItem(
                document=text,
                tier=strategy.obs_tier,
                memory_type=MemoryType.OBSERVATION,
                robot_id="bench",
            )
            manager.add(mem_item)
            if (i + 1) % consolidate_interval == 0:
                engine.run_all(llm=llm)

        engine.run_all(llm=llm)

        # Run all queries
        precisions = []
        judge_scores = []
        latencies = []

        for q in queries:
            query = q["query"]
            ground_truth = q["ground_truth"]
            token_budget = q["token_budget"]

            t0 = time.perf_counter()
            if strategy.dump_all:
                context = _dump_all(manager, strategy.inject_tiers, token_budget)
            else:
                context = injector.inject(_BENCH_STAGE, query, max_tokens=token_budget)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0)
            hits = _count_hits(context, ground_truth)
            precision = hits / len(ground_truth) if ground_truth else 0.0
            precisions.append(precision)

            if llm is not None:
                from benchmarks.llm_judge import llm_judge_score
                score = llm_judge_score(llm, query, context, ground_truth)
                judge_scores.append(score)

        store_size = manager.count()

    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else None

    return ScalingResult(
        strategy_name=strategy.name,
        scenario_name=scenario["name"],
        mode="",
        scale_param=0.0,
        store_size=store_size,
        avg_precision=avg_precision,
        avg_llm_judge_score=avg_judge,
        avg_injection_latency_ms=avg_latency,
    )


def run_replay_fraction(
    scenario_path: str | Path,
    strategy: StrategyConfig,
    fractions: list[float] | None = None,
    llm: LLMCallable | None = None,
) -> list[ScalingResult]:
    """Mode A: feed first N% of events, then run all queries."""
    if fractions is None:
        fractions = [0.25, 0.5, 0.75, 1.0]

    scenario = load_scenario(scenario_path)
    all_events = sorted(scenario["events"], key=lambda e: e["time"])

    results = []
    for frac in fractions:
        n = max(1, int(len(all_events) * frac))
        events = all_events[:n]

        result = _run_with_events(scenario, events, strategy, llm=llm)
        result.mode = "replay_fraction"
        result.scale_param = frac
        results.append(result)

    return results


def run_noise_padding(
    scenario_path: str | Path,
    strategy: StrategyConfig,
    distractor_counts: list[int] | None = None,
    llm: LLMCallable | None = None,
) -> list[ScalingResult]:
    """Mode B: feed all events + N distractor memories, then run queries."""
    if distractor_counts is None:
        distractor_counts = [0, 50, 200, 500]

    scenario = load_scenario(scenario_path)
    all_events = sorted(scenario["events"], key=lambda e: e["time"])
    max_time = all_events[-1]["time"] if all_events else 0

    results = []
    for count in distractor_counts:
        distractors = []
        for i in range(count):
            distractors.append({
                "time": max_time + 1 + i,
                "data": {"text": DISTRACTOR_POOL[i % len(DISTRACTOR_POOL)]},
            })

        events = all_events + distractors
        result = _run_with_events(scenario, events, strategy, llm=llm)
        result.mode = "noise_padding"
        result.scale_param = count
        results.append(result)

    return results


def print_scaling_results(results: list[ScalingResult]) -> None:
    """Pretty-print scaling results."""
    has_judge = any(r.avg_llm_judge_score is not None for r in results)
    header = (
        f"{'Strategy':<20} {'Mode':<18} {'Param':>7} {'Size':>6} "
        f"{'AvgPrec':>8} {'AvgLat(ms)':>11}"
    )
    if has_judge:
        header += f" {'AvgJudge':>9}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        param = f"{r.scale_param:.0%}" if r.mode == "replay_fraction" else f"{int(r.scale_param)}"
        line = (
            f"{r.strategy_name:<20} {r.mode:<18} {param:>7} {r.store_size:>6} "
            f"{r.avg_precision:>8.2f} {r.avg_injection_latency_ms:>11.2f}"
        )
        if has_judge:
            js = f"{r.avg_llm_judge_score:.2f}" if r.avg_llm_judge_score is not None else "N/A"
            line += f" {js:>9}"
        print(line)
    print("=" * len(header))


def scaling_results_to_json(results: list[ScalingResult]) -> list[dict]:
    """Convert scaling results to JSON-serializable dicts."""
    return [
        {
            "strategy": r.strategy_name,
            "scenario": r.scenario_name,
            "mode": r.mode,
            "scale_param": r.scale_param,
            "store_size": r.store_size,
            "avg_precision": round(r.avg_precision, 4),
            "avg_llm_judge_score": round(r.avg_llm_judge_score, 4) if r.avg_llm_judge_score is not None else None,
            "avg_injection_latency_ms": round(r.avg_injection_latency_ms, 2),
        }
        for r in results
    ]


QUICK_SCENARIOS = {"building_security.yaml"}
QUICK_STRATEGIES = {"full_context", "tiered_memory", "rag_with_recency"}
QUICK_FRACTIONS = [0.25, 1.0]
QUICK_DISTRACTOR_COUNTS = [0, 500]


if __name__ == "__main__":
    import json
    import sys

    quick = "--quick" in sys.argv

    llm = None
    if "--llm" in sys.argv:
        from typemem.llm import make_anthropic_llm
        llm = make_anthropic_llm()
        print("Using LLM judge (Haiku 4.5) for evaluation")

    scenario_dir = Path(__file__).parent / "scenarios"
    strategies = _default_strategies()
    if quick:
        strategies = [s for s in strategies if s.name in QUICK_STRATEGIES]

    scenario_files = sorted(scenario_dir.glob("*.yaml"))
    if quick:
        scenario_files = [p for p in scenario_files if p.name in QUICK_SCENARIOS]

    fractions = QUICK_FRACTIONS if quick else None
    distractor_counts = QUICK_DISTRACTOR_COUNTS if quick else None

    if quick:
        n_runs = len(scenario_files) * len(strategies) * (len(fractions) + len(distractor_counts))
        print(f"Quick mode: {len(scenario_files)} scenarios x {len(strategies)} strategies x {len(fractions) + len(distractor_counts)} scale points = {n_runs} runs")

    all_results: list[ScalingResult] = []

    for scenario_path in scenario_files:
        for strategy in strategies:
            all_results.extend(run_replay_fraction(
                scenario_path, strategy, fractions=fractions, llm=llm,
            ))
            all_results.extend(run_noise_padding(
                scenario_path, strategy, distractor_counts=distractor_counts, llm=llm,
            ))

    print_scaling_results(all_results)

    if "--json" in sys.argv:
        out_path = Path("scaling_results.json")
        with open(out_path, "w") as f:
            json.dump(scaling_results_to_json(all_results), f, indent=2)
        print(f"\nResults saved to {out_path}")
