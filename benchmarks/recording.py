"""End-to-end benchmark: replay a TypeGo recording into typemem and score retrieval.

Pipeline
--------
1. Load a recording directory (``events.jsonl`` from TypeGo's EventRecorder).
2. Replay the events through the live event bus into a fresh ``MemoryManager``,
   driven by ``SceneObserver`` (in-process structural dedup by ``(name, waypoint)``).
3. For each strategy in :func:`benchmarks.synthetic._default_strategies`, run a
   small set of hand-authored "where did you see X?" queries via ``MemoryInjector``.
4. Score each (strategy, query) pair with the LLM judge if ``--llm`` is passed,
   plus the substring-precision metric used by the synthetic benchmark.

This intentionally exercises the full live pipeline: recording → ``EventReplay``
→ ``events`` bus → ``SceneObserver`` → ``MemoryManager`` → ``ConsolidationEngine``
→ ``MemoryInjector``. Approach (1) of the design — pre-converting the recording
to a YAML scenario — would skip everything from "events bus" onward.

Usage::

    python -m benchmarks.recording RECORDING_DIR              # substring eval only
    python -m benchmarks.recording RECORDING_DIR --llm        # adds LLM judge
    python -m benchmarks.recording RECORDING_DIR --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from benchmarks.synthetic import (
    BenchmarkResult,
    QueryResult,
    StrategyConfig,
    _BENCH_STAGE,
    _default_strategies,
    _dump_all,
    _count_hits,
    print_results,
    results_to_json,
)
from typemem import events
from typemem.consolidation import ConsolidationEngine
from typemem.injector import MemoryInjector, StageConfig
from typemem.llm import LLMCallable
from typemem.memory_manager import MemoryManager
from typemem.plugins.scene_observer import SceneObserver
from typemem.replay import EventReplay


# Hand-authored queries chosen to cover both single-waypoint and multi-waypoint
# objects in recordings/20260406_145037. The ground_truth strings are derived
# automatically below from the actual contents of the JSONL — see
# `_build_ground_truth`. Each entry only declares the *object name* to ask about.
_QUERY_TEMPLATES: List[Tuple[str, str]] = [
    ("dog",          "Where did you see the dog?"),
    ("refrigerator", "Where did you see the refrigerator?"),
    ("bed",          "Where did you see the bed?"),
    ("teddy bear",   "Where did you see the teddy bear?"),
    ("clock",        "Where did you see the clock?"),
    ("chair",        "Where did you see chairs?"),
]

# Single token budget for all queries — the recording is short and we
# don't need budget sweeps for the smoke test.
_TOKEN_BUDGET = 400


@dataclass
class RecordingScenarioSpec:
    """A scenario derived from a real recording: queries + computed ground truth."""
    name: str
    recording_dir: str
    queries: List[Dict]  # each: {"query": str, "ground_truth": List[str], "object": str}


def _build_ground_truth(recording_dir: str) -> Dict[str, List[int]]:
    """Scan the JSONL once and return {object_name: sorted_unique_waypoints}."""
    out: Dict[str, set] = defaultdict(set)
    jsonl = Path(recording_dir) / "events.jsonl"
    with open(jsonl) as f:
        for line in f:
            ev = json.loads(line)
            if ev.get("ch") != "scene_object":
                continue
            data = ev.get("data", {})
            name = data.get("name")
            waypoint = data.get("waypoint")
            if name is None or waypoint is None:
                continue
            out[name].add(int(waypoint))
    return {k: sorted(v) for k, v in out.items()}


def build_scenario(recording_dir: str, name: Optional[str] = None) -> RecordingScenarioSpec:
    """Construct queries + ground truth from a recording.

    Skips template queries whose object never actually appears in this recording —
    so the same template list works for any TypeGo recording without crashing.
    """
    object_waypoints = _build_ground_truth(recording_dir)
    queries: List[Dict] = []
    for obj, query_text in _QUERY_TEMPLATES:
        wps = object_waypoints.get(obj)
        if not wps:
            continue
        ground_truth = [f"waypoint {wp}" for wp in wps]
        queries.append({
            "object": obj,
            "query": query_text,
            "ground_truth": ground_truth,
        })

    return RecordingScenarioSpec(
        name=name or Path(recording_dir).name,
        recording_dir=recording_dir,
        queries=queries,
    )


def _populate_memory(
    recording_dir: str,
    manager: MemoryManager,
    robot_id: str,
) -> Tuple[int, int]:
    """Drive the recording through the live event bus → SceneObserver → manager.

    Returns
    -------
    (events_replayed, items_emitted)
    """
    events.reset()  # avoid leftover subscribers from previous runs

    observer = SceneObserver()
    observer.setup(manager, robot_id)

    replay = EventReplay(recording_dir)
    n_replayed = replay.play(speed=0)
    emitted = observer.run()
    return n_replayed, len(emitted)


def run_recording_benchmark(
    spec: RecordingScenarioSpec,
    strategy: StrategyConfig,
    llm: Optional[LLMCallable] = None,
) -> BenchmarkResult:
    """Run one strategy against one recording-derived scenario."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MemoryManager(persist_dir=tmpdir, robot_id="bench")
        engine = ConsolidationEngine(manager, expiry_interval=9999)
        for plugin in strategy.make_fresh_plugins():
            engine.register_strategy(plugin)

        injector = MemoryInjector(manager, cache_ttl=0)
        injector.set_stage_config(_BENCH_STAGE, StageConfig(
            tiers=strategy.inject_tiers,
            max_tokens=_TOKEN_BUDGET,
            n_results=strategy.inject_n_results,
            recency_weight=strategy.inject_recency_weight,
        ))

        n_replayed, n_items = _populate_memory(spec.recording_dir, manager, "bench")

        # Run any registered consolidation once on the populated store.
        engine.run_all(llm=llm)

        query_results: List[QueryResult] = []
        for q in spec.queries:
            query = q["query"]
            ground_truth = q["ground_truth"]

            t0 = time.perf_counter()
            if strategy.dump_all:
                context = _dump_all(manager, strategy.inject_tiers, _TOKEN_BUDGET)
            else:
                context = injector.inject(_BENCH_STAGE, query, max_tokens=_TOKEN_BUDGET)
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
                time=0,
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
        scenario_name=f"{spec.name} ({n_replayed} ev → {n_items} items)",
        query_results=query_results,
        total_memories=total_memories,
        avg_precision=avg_precision,
        avg_injection_latency_ms=avg_injection_latency_ms,
        avg_token_count=avg_token_count,
        avg_llm_judge_score=avg_llm_judge_score,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_dir", help="Path to a TypeGo recording directory")
    parser.add_argument("--llm", action="store_true", help="Score with LLM judge (Haiku)")
    parser.add_argument("--json", metavar="OUT", help="Also write results JSON to OUT")
    parser.add_argument("--strategy", help="Run only this strategy by name")
    args = parser.parse_args(argv)

    rec_dir = Path(args.recording_dir).expanduser()
    if not (rec_dir / "events.jsonl").is_file():
        print(f"error: no events.jsonl in {rec_dir}", file=sys.stderr)
        return 2

    spec = build_scenario(str(rec_dir))
    if not spec.queries:
        print(f"error: no template-query objects appear in {rec_dir}", file=sys.stderr)
        return 2

    print(f"Recording: {rec_dir}")
    print(f"Queries derived from recording content:")
    for q in spec.queries:
        print(f"  {q['query']!r}  ground_truth={q['ground_truth']}")
    print()

    llm: Optional[LLMCallable] = None
    if args.llm:
        from typemem.llm import make_anthropic_llm
        llm = make_anthropic_llm()
        print("Using LLM judge (Haiku 4.5) for evaluation\n")

    strategies = _default_strategies()
    if args.strategy:
        strategies = [s for s in strategies if s.name == args.strategy]
        if not strategies:
            print(f"error: unknown strategy {args.strategy!r}", file=sys.stderr)
            return 2

    results: List[BenchmarkResult] = []
    for strategy in strategies:
        result = run_recording_benchmark(spec, strategy, llm=llm)
        results.append(result)

    print_results(results)

    if args.json:
        out_path = Path(args.json)
        with open(out_path, "w") as f:
            json.dump(results_to_json(results), f, indent=2)
        print(f"\nResults saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
