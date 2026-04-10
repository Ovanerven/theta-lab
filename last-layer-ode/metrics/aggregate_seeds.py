"""Aggregate multi-seed CSV results from compare_runs.py.

Takes N CSVs (one per seed run) and outputs a single CSV with
mean and std per run × species column.

Usage:
    python last-layer-ode/metrics/aggregate_seeds.py \\
        results/ablation_s42.csv results/ablation_s123.csv results/ablation_s456.csv \\
        --out results/ablation_aggregated.csv

Each input CSV must have the format produced by compare_runs.py:
    run, nrmse, <species1>, <species2>, ...
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def aggregate(all_rows: list[list[dict]]) -> list[dict]:
    """
    all_rows: one list of dicts per seed CSV.
    Returns one dict per run with mean and std of nrmse and each species column.
    """
    # Collect all run names and columns from the first CSV
    runs   = [r["run"] for r in all_rows[0]]
    cols   = [c for c in all_rows[0][0] if c != "run"]

    result = []
    for run in runs:
        values: dict[str, list[float]] = {c: [] for c in cols}
        for seed_rows in all_rows:
            row = next((r for r in seed_rows if r["run"] == run), None)
            if row is None:
                continue
            for c in cols:
                if row[c] != "":
                    values[c].append(float(row[c]))

        out: dict = {"run": run}
        for c in cols:
            v = np.array(values[c])
            out[f"{c}_mean"] = float(np.mean(v)) if len(v) else float("nan")
            out[f"{c}_std"]  = float(np.std(v))  if len(v) else float("nan")
            out[f"{c}_n"]    = len(v)
        result.append(out)

    return sorted(result, key=lambda r: r["nrmse_mean"])


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        print("No rows to save.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows({k: (f"{v:.6f}" if isinstance(v, float) else v)
                          for k, v in r.items()} for r in rows)
    print(f"Saved {path}")


def print_table(rows: list[dict]) -> None:
    if not rows:
        return
    mean_cols = [c for c in rows[0] if c.endswith("_mean")]
    name_w = max(len(r["run"]) for r in rows)
    header = f"{'run':<{name_w}}" + "".join(f"  {c[:-5]:>12}" for c in mean_cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        n = r.get("nrmse_n", "?")
        line = f"{r['run']:<{name_w}}"
        for c in mean_cols:
            mean = r[c]
            std  = r.get(c[:-5] + "_std", float("nan"))
            line += f"  {mean:>7.4f}±{std:.4f}"
        print(f"{line}  (n={n})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+", help="Per-seed CSV files from compare_runs.py")
    parser.add_argument("--out", type=str, required=True, help="Output aggregated CSV path")
    args = parser.parse_args()

    all_rows = [load_csv(Path(p)) for p in args.csvs]
    print(f"Loaded {len(all_rows)} seed CSVs, {len(all_rows[0])} runs each")

    aggregated = aggregate(all_rows)
    print_table(aggregated)
    save_csv(aggregated, Path(args.out))
