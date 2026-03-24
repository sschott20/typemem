"""Publication-ready plots for typemem benchmark results.

Usage:
    python -m benchmarks.plot --synthetic benchmark_results.json --output-dir figures/
    python -m benchmarks.plot --scaling scaling_results.json --output-dir figures/
    python -m benchmarks.plot --latency latency_results.json --output-dir figures/
    python -m benchmarks.plot --all --output-dir figures/
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.plot_utils import (
    apply_style, save_fig, get_color, get_label, get_linestyle, get_marker,
    FIG_WIDTH, FIG_HEIGHT,
)


def plot_strategy_comparison(data: list[dict], output_dir: str) -> None:
    """Plot 1: Grouped bar chart of avg precision per strategy, grouped by scenario."""
    apply_style()
    os.makedirs(output_dir, exist_ok=True)

    by_scenario: dict[str, dict[str, float]] = defaultdict(dict)
    for entry in data:
        by_scenario[entry["scenario"]][entry["strategy"]] = entry["avg_precision"]

    scenarios = sorted(by_scenario.keys())
    strategies = sorted({s for sc in by_scenario.values() for s in sc})

    n_scenarios = len(scenarios)
    n_strategies = len(strategies)
    x = np.arange(n_scenarios)
    bar_width = 0.8 / n_strategies

    fig, ax = plt.subplots(figsize=(max(FIG_WIDTH, n_scenarios * 1.2), FIG_HEIGHT))

    for i, strat in enumerate(strategies):
        values = [by_scenario[sc].get(strat, 0.0) for sc in scenarios]
        offset = (i - n_strategies / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width * 0.9,
               label=get_label(strat), color=get_color(strat))

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Avg Precision")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios],
                       rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", framealpha=0.9)

    save_fig(fig, os.path.join(output_dir, "strategy_comparison"))


def plot_quality_vs_size(data: list[dict], output_dir: str, mode: str = "noise_padding") -> None:
    """Plot 2: Quality score vs store size, one line per strategy."""
    apply_style()
    os.makedirs(output_dir, exist_ok=True)

    filtered = [d for d in data if d["mode"] == mode]
    if not filtered:
        return

    by_strategy: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for entry in filtered:
        metric = entry.get("avg_llm_judge_score") or entry["avg_precision"]
        by_strategy[entry["strategy"]].append((entry["store_size"], metric))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    for strat in sorted(by_strategy):
        points = sorted(by_strategy[strat])
        sizes = [p[0] for p in points]
        scores = [p[1] for p in points]
        ax.plot(sizes, scores,
                color=get_color(strat),
                linestyle=get_linestyle(strat),
                marker=get_marker(strat),
                markersize=4,
                label=get_label(strat))

    ax.set_xlabel("Store Size")
    ax.set_ylabel("Quality Score")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", framealpha=0.9)

    save_fig(fig, os.path.join(output_dir, "quality_vs_size"))


def plot_latency_vs_size(data: list[dict], output_dir: str) -> None:
    """Plot 3: Injection latency (p50) vs store size, one line per strategy."""
    apply_style()
    os.makedirs(output_dir, exist_ok=True)

    by_strategy: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for entry in data:
        by_strategy[entry["strategy"]].append((entry["store_size"], entry["inj_p50"]))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    for strat in sorted(by_strategy):
        points = sorted(by_strategy[strat])
        sizes = [p[0] for p in points]
        latencies = [p[1] for p in points]
        ax.plot(sizes, latencies,
                color=get_color(strat),
                linestyle=get_linestyle(strat),
                marker=get_marker(strat),
                markersize=4,
                label=get_label(strat))

    ax.set_xlabel("Store Size")
    ax.set_ylabel("Injection Latency p50 (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper left", framealpha=0.9)

    save_fig(fig, os.path.join(output_dir, "latency_vs_size"))


def plot_latency_distribution(data: list[dict], output_dir: str, store_size: int | None = None) -> None:
    """Plot 4: Box plot of injection latency distributions per strategy.

    If store_size is given, filter to that size only. Otherwise use the largest
    store size available per strategy.
    """
    apply_style()
    os.makedirs(output_dir, exist_ok=True)

    best: dict[str, dict] = {}
    for entry in data:
        strat = entry["strategy"]
        if store_size is not None and entry["store_size"] != store_size:
            continue
        if strat not in best or entry["store_size"] > best[strat]["store_size"]:
            best[strat] = entry

    if not best:
        return

    strategies = sorted(best.keys())
    all_latencies = [best[s]["injection_latencies_ms"] for s in strategies]
    colors = [get_color(s) for s in strategies]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    bp = ax.boxplot(all_latencies, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels([get_label(s) for s in strategies], rotation=30, ha="right")
    ax.set_ylabel("Injection Latency (ms)")

    save_fig(fig, os.path.join(output_dir, "latency_distribution"))
