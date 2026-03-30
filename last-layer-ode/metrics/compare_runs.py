"""Compare all runs in an experiment folder by NRMSE.

Computes NRMSE on the test split of each run (or loads cached results).
Cache is saved to <exp_root>/nrmse_cache.csv.

Usage:
    python last-layer-ode/metrics/compare_runs.py experiments/scaffold_size_effect
    python last-layer-ode/metrics/compare_runs.py experiments/scaffold_size_effect --recompute
    python last-layer-ode/metrics/compare_runs.py experiments/scaffold_size_effect --csv results/scaffold.csv
    python last-layer-ode/metrics/compare_runs.py experiments/scaffold_size_effect --plot
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.nrmse import load_or_compute


def _aggregate(rows: list[dict]) -> list[dict]:
    """One row per run: mean of per-species medians as the sort key."""
    by_run: dict[str, dict] = {}
    for r in rows:
        run = r["run"]
        if run not in by_run:
            by_run[run] = {"run": run, "scaffold": r["scaffold"],
                           "P": r["P"], "species": {}}
        by_run[run]["species"][r["species"]] = r["median"]

    result = []
    for run, d in by_run.items():
        d["primary"] = float(np.mean(list(d["species"].values())))
        result.append(d)
    return sorted(result, key=lambda r: r["primary"])


def print_table(runs: list[dict]) -> None:
    if not runs:
        print("No completed runs found.")
        return

    all_species: list[str] = []
    for r in runs:
        for s in r["species"]:
            if s not in all_species:
                all_species.append(s)

    col_w, sp_w = 55, 10
    header = f"{'run':<{col_w}}  {'NRMSE':>8}"
    for s in all_species:
        header += f"  {s[:sp_w]:>{sp_w}}"
    print(header)
    print("-" * len(header))

    for r in runs:
        line = f"{r['run']:<{col_w}}  {r['primary']:>8.4f}"
        for s in all_species:
            val = r["species"].get(s)
            line += f"  {val:>{sp_w}.4f}" if val is not None else f"  {'N/A':>{sp_w}}"
        print(line)


def save_csv(runs: list[dict], path: Path) -> None:
    all_species: list[str] = []
    for r in runs:
        for s in r["species"]:
            if s not in all_species:
                all_species.append(s)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "nrmse"] + all_species)
        for r in runs:
            name = re.sub(r"^\d{8}_\d{6}_", "", r["run"])
            writer.writerow(
                [name, f"{r['primary']:.6f}"]
                + [f"{r['species'].get(s, ''):.6f}" if r["species"].get(s) is not None else ""
                   for s in all_species]
            )
    print(f"Saved CSV to {path}")


def plot_comparison(runs: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["run"] for r in runs]
    values = [r["primary"] for r in runs]

    fig, ax = plt.subplots(figsize=(max(8, len(runs) * 0.8), 5))
    bars = ax.bar(range(len(names)), values)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("NRMSE (lower is better)")
    ax.set_title("Run comparison")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Experiment folder, e.g. experiments/scaffold_size_effect")
    parser.add_argument("--recompute", action="store_true", help="Ignore cache and recompute NRMSE")
    parser.add_argument("--csv", type=str, default=None, help="Save summary CSV to this path")
    parser.add_argument("--plot", action="store_true", help="Save a bar chart")
    parser.add_argument("--plot-out", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    rows = load_or_compute(root, recompute=args.recompute)
    runs = _aggregate(rows)
    print_table(runs)

    if args.csv:
        save_csv(runs, Path(args.csv))
    if args.plot:
        out = Path(args.plot_out) if args.plot_out else root / "comparison.png"
        plot_comparison(runs, out)
