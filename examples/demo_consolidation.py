"""Demo: run tiered_memory on kitchen_patrol with live web visualization."""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from typemem.consolidation import ConsolidationEngine
from typemem.injector import MemoryInjector, StageConfig
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.plugins.text_summary import TextSummaryPlugin
from typemem.viz import start_viz


def load_scenario(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_consolidation_with_viz(engine, manager, viz):
    """Run consolidation and notify viz with actual input/output data."""
    m1_before = {m.id: m for m in manager.get_by_tier(MemoryTier.M1)}
    processed_before = set(engine.processed_index._data.get("text_summary", set()))

    t0 = time.perf_counter()
    results = engine.run_all()
    dur = (time.perf_counter() - t0) * 1000

    processed_after = set(engine.processed_index._data.get("text_summary", set()))
    newly_processed = processed_after - processed_before

    for name, ids in results.items():
        if ids:
            inputs = [
                {"id": mid, "text": m1_before[mid].document, "tier": "M1"}
                for mid in newly_processed if mid in m1_before
            ]
            outputs = []
            for oid in ids:
                out_item = manager.get(oid)
                if out_item:
                    outputs.append({"id": oid, "text": out_item.document, "tier": out_item.tier.label})
            viz.notify_consolidation(name, inputs, outputs, duration_ms=dur)
            return name, len(inputs), len(outputs)
    return None, 0, 0


def main():
    scenario = load_scenario("benchmarks/scenarios/kitchen_patrol.yaml")
    events = sorted(scenario["events"], key=lambda e: e["time"])
    queries = sorted(scenario["queries"], key=lambda q: q["time"])

    timeline = []
    for e in events:
        timeline.append({"type": "event", "time": e["time"], "data": e["data"]})
    for q in queries:
        timeline.append({
            "type": "query", "time": q["time"],
            "query": q["query"], "ground_truth": q["ground_truth"],
            "token_budget": q["token_budget"],
        })
    timeline.sort(key=lambda x: (x["time"], 0 if x["type"] == "event" else 1))

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MemoryManager(persist_dir=tmpdir, robot_id="demo")
        engine = ConsolidationEngine(manager, expiry_interval=9999)
        engine.register_strategy(TextSummaryPlugin(batch_size=3, interval=0))
        injector = MemoryInjector(manager, cache_ttl=0)
        injector.set_stage_config("bench", StageConfig(
            tiers=[MemoryTier.M1, MemoryTier.M2],
            max_tokens=1000, n_results=10, recency_weight=0.3,
        ))

        # Start viz server
        viz = start_viz(manager, engine=engine, injector=injector,
                        port=8811, open_browser=True)
        print(f"\n  Viz running at http://localhost:{viz.port}")
        print(f"  Scenario: {scenario['name']}")
        print(f"  Events: {len(events)}  Queries: {len(queries)}")
        print(f"  Strategy: tiered_memory (TextSummaryPlugin, batch_size=3)\n")

        # Small delay so browser can connect SSE before events start
        time.sleep(1.0)

        events_fed = 0
        for item in timeline:
            if item["type"] == "event":
                text = item["data"].get("text", str(item["data"]))
                mem = MemoryItem(
                    document=text, tier=MemoryTier.M1,
                    memory_type=MemoryType.OBSERVATION, robot_id="demo",
                )
                manager.add(mem)
                events_fed += 1
                viz.notify_add(mem)
                print(f"  t={item['time']:>4}s  [EVENT] {text}")

                if events_fed % 3 == 0:
                    name, n_in, n_out = run_consolidation_with_viz(engine, manager, viz)
                    if name:
                        print(f"          [CONSOLIDATION] {name}: {n_in} in → {n_out} out")

                # Pace events so you can watch in the browser
                time.sleep(0.5)
            else:
                run_consolidation_with_viz(engine, manager, viz)

                t0 = time.perf_counter()
                context = injector.inject("bench", item["query"], max_tokens=item["token_budget"])
                dur = (time.perf_counter() - t0) * 1000
                hits = sum(1 for gt in item["ground_truth"] if gt.lower() in context.lower())
                total = len(item["ground_truth"])

                viz.notify_injection("bench", item["query"], context,
                                     injector.last_results, duration_ms=dur)

                print(f"\n  t={item['time']:>4}s  [QUERY] \"{item['query']}\"")
                print(f"          hits={hits}/{total}  precision={hits/total:.2f}")
                print(f"          context:")
                for line in context.split("\n"):
                    print(f"            {line}")
                print()
                time.sleep(0.5)

        print(f"\n  Done! {manager.count()} memories in store.")
        print(f"  Viz still running at http://localhost:{viz.port}")
        print(f"  Press Ctrl+C to stop.\n")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            viz.stop()
            print("  Stopped.")


if __name__ == "__main__":
    main()
