# typemem/demo.py
"""End-to-end demo for typemem memory strategies.

Usage:
    python -m typemem.demo                  # narrated simulation (default)
    python -m typemem.demo --compare        # metrics comparison table
    python -m typemem.demo --interactive    # interactive REPL
    python -m typemem.demo --all            # all three modes
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import yaml

from typemem.baselines import make_full_context, make_monolithic_rag
from typemem.chromadb_store import ChromaDBStore
from typemem.tiered import make_tiered_llm

_DEFAULT_SCENARIO = (
    Path(__file__).parent.parent / "benchmarks" / "scenarios" / "building_security.yaml"
)

# Strategy configs: (display_name, factory_fn, injection_name)
_STRATEGIES = [
    ("full_context", make_full_context, "dump"),
    ("monolithic_rag", make_monolithic_rag, "topk"),
    ("tiered_llm", make_tiered_llm, "tiered"),
]


def _load_scenario(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_timeline(scenario: dict) -> list[dict]:
    """Merge events and queries into a single sorted timeline."""
    timeline = []
    for e in scenario["events"]:
        timeline.append({"type": "event", "time": e["time"], "data": e["data"]})
    for q in scenario["queries"]:
        timeline.append({
            "type": "query", "time": q["time"], "query": q["query"],
            "ground_truth": q["ground_truth"], "token_budget": q["token_budget"],
        })
    timeline.sort(key=lambda x: (x["time"], 0 if x["type"] == "event" else 1))
    return timeline


# ---------------------------------------------------------------------------
# Mode 1: Narrated simulation
# ---------------------------------------------------------------------------

def run_narrate(scenario_path: Path, sleep_per_event: float = 0.03):
    """Step through the timeline printing events, consolidation, and queries."""
    scenario = _load_scenario(scenario_path)
    timeline = _build_timeline(scenario)

    # Create one system per strategy
    tmpdirs = []
    stores = {}
    systems = {}
    injection_names = {}
    for name, factory, inj_name in _STRATEGIES:
        td = tempfile.mkdtemp()
        tmpdirs.append(td)
        stores[name] = ChromaDBStore(persist_dir=td)
        systems[name] = factory(stores[name])
        injection_names[name] = inj_name

    events_fed = 0
    print(f"\n{'=' * 70}")
    print(f"  {scenario['name']}")
    print(f"{'=' * 70}\n")

    for item in timeline:
        if item["type"] == "event":
            text = item["data"].get("text", "")
            # Highlight anomalies
            is_anomaly = any(w in text.lower() for w in [
                "unlocked", "unfamiliar", "alert", "emergency", "unbadged",
            ])
            marker = " (!)" if is_anomaly else ""
            print(f"  [t={item['time']:>5}s]{marker} {text}")

            for sys in systems.values():
                sys.observe(item["data"])
            events_fed += 1

            # Consolidate every 10 events
            if events_fed % 10 == 0:
                for name, sys in systems.items():
                    created = sys.consolidate()
                    if created and name == "tiered_llm":
                        for cid in created:
                            entry = stores[name].get(cid)
                            if entry:
                                tier = entry.metadata.get("_tier", "?")
                                print(f"           >> [{tier.upper()}] {entry.text}")

            time.sleep(sleep_per_event)

        else:
            # Query point — consolidate all, then show side-by-side
            for sys in systems.values():
                sys.consolidate()

            print(f"\n  {'~' * 60}")
            print(f"  [t={item['time']:>5}s] QUERY: \"{item['query']}\"")
            print(f"  {'~' * 60}")

            for name in ["full_context", "monolithic_rag", "tiered_llm"]:
                context = systems[name].inject(
                    injection_names[name], item["query"], item["token_budget"],
                )
                tokens = len(context) // 4
                lines = context.strip().split("\n") if context.strip() else []
                preview = lines[:5]

                print(f"\n    {name.upper()} ({tokens} tokens, {len(lines)} lines):")
                for line in preview:
                    print(f"      {line[:100]}")
                if len(lines) > 5:
                    print(f"      ... ({len(lines) - 5} more lines)")

            print()


# ---------------------------------------------------------------------------
# Mode 2: Comparison table
# ---------------------------------------------------------------------------

def run_compare(scenario_path: Path):
    """Run all strategies and print a metrics comparison."""
    scenario = _load_scenario(scenario_path)
    timeline = _build_timeline(scenario)

    results = []
    for strategy_name, factory, injection_name in _STRATEGIES:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChromaDBStore(persist_dir=tmpdir)
            system = factory(store)
            query_results = _run_strategy(timeline, system, store, injection_name)
            total = store.count()

            avg_prec = (
                sum(qr["precision"] for qr in query_results) / len(query_results)
                if query_results else 0.0
            )
            avg_lat = (
                sum(qr["latency_ms"] for qr in query_results) / len(query_results)
                if query_results else 0.0
            )
            avg_tok = (
                sum(qr["tokens"] for qr in query_results) / len(query_results)
                if query_results else 0.0
            )
            results.append({
                "name": strategy_name, "memories": total,
                "avg_precision": avg_prec, "avg_latency_ms": avg_lat,
                "avg_tokens": avg_tok, "queries": query_results,
            })

    # Print table
    header = (
        f"{'Strategy':<20} {'Memories':>8} {'AvgPrec':>8} "
        f"{'AvgLat(ms)':>11} {'AvgTokens':>10}"
    )
    print(f"\n{'=' * len(header)}")
    print(f"  {scenario['name']}")
    print(f"{'=' * len(header)}")
    print(header)
    print(f"{'=' * len(header)}")
    for r in results:
        print(
            f"{r['name']:<20} {r['memories']:>8} {r['avg_precision']:>8.2f} "
            f"{r['avg_latency_ms']:>11.2f} {r['avg_tokens']:>10.1f}"
        )
    print(f"{'=' * len(header)}")

    # Per-query detail
    for r in results:
        print(f"\n--- {r['name']} ---")
        for qr in r["queries"]:
            print(
                f"  t={qr['time']:>5}s  query={qr['query']!r:<55} "
                f"hits={qr['hits']}/{qr['total_gt']}  "
                f"prec={qr['precision']:.2f}  "
                f"tokens={qr['tokens']}"
            )


def _run_strategy(timeline, system, store, injection_name):
    """Run a strategy through the timeline, return per-query results."""
    query_results = []
    events_fed = 0

    for item in timeline:
        if item["type"] == "event":
            system.observe(item["data"])
            events_fed += 1
            if events_fed % 10 == 0:
                system.consolidate()
        else:
            system.consolidate()
            query = item["query"]
            gt = item["ground_truth"]
            budget = item["token_budget"]

            t0 = time.perf_counter()
            context = system.inject(injection_name, query, budget)
            latency_ms = (time.perf_counter() - t0) * 1000

            ctx_lower = context.lower()
            hits = sum(1 for g in gt if g.lower() in ctx_lower)
            precision = hits / len(gt) if gt else 0.0
            tokens = len(context) // 4

            query_results.append({
                "time": item["time"], "query": query,
                "hits": hits, "total_gt": len(gt),
                "precision": precision, "tokens": tokens,
                "latency_ms": latency_ms,
            })
    return query_results


# ---------------------------------------------------------------------------
# Mode 3: Interactive REPL
# ---------------------------------------------------------------------------

def run_interactive(scenario_path: Path):
    """Load scenario with tiered memory, then enter interactive query mode."""
    scenario = _load_scenario(scenario_path)
    events = sorted(scenario["events"], key=lambda e: e["time"])

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaDBStore(persist_dir=tmpdir)
        system = make_tiered_llm(store)

        # Feed all events
        print(f"\nLoading {len(events)} events...", end="", flush=True)
        for i, event in enumerate(events):
            system.observe(event["data"])
            if (i + 1) % 10 == 0:
                system.consolidate()
        system.consolidate()
        print(" done.")

        # Stats
        total = store.count()
        raw = store.count(filters={"_tier": "raw"})
        summary = store.count(filters={"_tier": "summary"})
        knowledge = store.count(filters={"_tier": "knowledge"})

        print(f"\n{'=' * 60}")
        print(f"  Memory: {total} total ({raw} raw, {summary} summary, {knowledge} knowledge)")
        print(f"{'=' * 60}")
        print("  Commands: type a query, 'dump [tier]', 'stats', or 'quit'\n")

        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                break

            if user_input.lower() == "stats":
                print(
                    f"  Total: {store.count()}, "
                    f"Raw: {store.count(filters={'_tier': 'raw'})}, "
                    f"Summary: {store.count(filters={'_tier': 'summary'})}, "
                    f"Knowledge: {store.count(filters={'_tier': 'knowledge'})}"
                )
                continue

            if user_input.lower().startswith("dump"):
                parts = user_input.split()
                tier_filter = parts[1] if len(parts) > 1 else None
                if tier_filter:
                    entries = store.get_all(filters={"_tier": tier_filter})
                else:
                    entries = store.get_all()
                for e in entries:
                    t = e.metadata.get("_tier", "?")
                    print(f"  [{t.upper()}] {e.text}")
                print(f"  ({len(entries)} entries)")
                continue

            # Treat as query
            t0 = time.perf_counter()
            context = system.inject("tiered", user_input, token_budget=400)
            elapsed = (time.perf_counter() - t0) * 1000
            tokens = len(context) // 4

            if context.strip():
                print(context)
            else:
                print("  (no relevant memories found)")
            print(f"  ({tokens} tokens, {elapsed:.1f}ms)\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="typemem demo: compare memory strategies on a patrol scenario",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--narrate", action="store_true", help="Narrated simulation (default)")
    mode.add_argument("--compare", action="store_true", help="Comparison metrics table")
    mode.add_argument("--interactive", action="store_true", help="Interactive query REPL")
    mode.add_argument("--all", action="store_true", help="Run all three modes in sequence")
    parser.add_argument("--scenario", type=str, default=None, help="Path to scenario YAML")
    args = parser.parse_args()

    scenario = Path(args.scenario) if args.scenario else _DEFAULT_SCENARIO
    if not scenario.exists():
        print(f"Error: scenario not found: {scenario}")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY not set.\n"
            "The tiered strategy requires LLM calls for consolidation.\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "Estimated cost per run: ~$0.02"
        )
        sys.exit(1)

    if args.all:
        run_narrate(scenario)
        print(f"\n{'=' * 70}\n")
        run_compare(scenario)
        print(f"\n{'=' * 70}\n")
        run_interactive(scenario)
    elif args.compare:
        run_compare(scenario)
    elif args.interactive:
        run_interactive(scenario)
    else:
        run_narrate(scenario)


if __name__ == "__main__":
    main()
