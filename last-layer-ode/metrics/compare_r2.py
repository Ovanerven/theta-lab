"""Compare endpoint R² across all runs in a study folder.

Computes R²(protein final) and R²(mRNA max) on the test split for each run,
caches results per run (r2_cache.csv), and prints a ranked table. When the
test split contains samples from multiple data sources (old/new), also
computes and persists per-source R² alongside the overall numbers.

Usage:
    python last-layer-ode/metrics/compare_r2.py experiments/txtl_supervisor_combined_sweep
    python last-layer-ode/metrics/compare_r2.py experiments/txtl_supervisor_combined_sweep --csv results/r2.csv
    python last-layer-ode/metrics/compare_r2.py experiments/txtl_supervisor_combined_sweep --plot
    python last-layer-ode/metrics/compare_r2.py experiments/txtl_supervisor_combined_sweep --recompute
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.endpoint_r2 import collect_endpoints, r2, r2_by_source, save_r2_cache
from plot_diagnostics import device_auto

CACHE_NAME = "r2_cache.csv"
# Required columns; per-source columns are added dynamically when present.
BASE_FIELDS = ["run", "n", "r2_protein_final", "r2_mrna_max"]
SOURCE_FIELDS = [
    "r2_protein_old", "r2_mrna_old", "n_old",
    "r2_protein_new", "r2_mrna_new", "n_new",
]


def _find_exp_dirs(root: Path) -> list[Path]:
    return sorted(
        d for d in root.iterdir()
        if d.is_dir()
        and (d / "config.yaml").exists()
        and (d / "model.pt").exists()
        and (d / "split.npz").exists()
    )


def _maybe_float(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _maybe_int(x):
    if x is None or x == "":
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _load_cache(exp_dir: Path) -> dict | None:
    cache = exp_dir / CACHE_NAME
    if not cache.exists():
        return None
    with open(cache, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[0]
    out: dict = {
        "run": row["run"],
        "n": int(row["n"]),
        "r2_protein_final": float(row["r2_protein_final"]),
        "r2_mrna_max": float(row["r2_mrna_max"]),
    }
    for k in SOURCE_FIELDS:
        if k in row:
            v = _maybe_int(row[k]) if k.startswith("n_") else _maybe_float(row[k])
            if v is not None:
                out[k] = v
    return out


def _compute_one(exp_dir: Path, device) -> dict:
    raw = collect_endpoints(exp_dir, device, split="test", protein_sp="pm", mrna_sp="mm")
    r2_p = float(r2(raw["true_protein_final"], raw["pred_protein_final"]))
    r2_m = float(r2(raw["true_mrna_max"],      raw["pred_mrna_max"]))
    # save_r2_cache already handles per-source columns when sources are present
    save_r2_cache(exp_dir, raw, r2_p, r2_m)
    result: dict = {
        "run": exp_dir.name,
        "n": int(raw["n"]),
        "r2_protein_final": r2_p,
        "r2_mrna_max": r2_m,
    }
    by_src = r2_by_source(raw)
    src_names = {0: "old", 1: "new"}
    for src_id, name in src_names.items():
        if src_id in by_src:
            s = by_src[src_id]
            result[f"r2_protein_{name}"] = float(s["r2_protein"])
            result[f"r2_mrna_{name}"]    = float(s["r2_mrna"])
            result[f"n_{name}"]           = int(s["n"])
    return result


def load_or_compute(root: Path, recompute: bool = False) -> list[dict]:
    exp_dirs = _find_exp_dirs(root)
    if not exp_dirs:
        print(f"No completed runs found in {root}")
        return []

    device = device_auto()
    results = []
    for exp_dir in exp_dirs:
        cached = None if recompute else _load_cache(exp_dir)
        if cached is not None:
            results.append(cached)
            continue

        print(f"  computing R² for {exp_dir.name} ...")
        try:
            results.append(_compute_one(exp_dir, device))
        except Exception as e:
            print(f"    skip ({exp_dir.name}): {e}")

    return sorted(results, key=lambda r: r["r2_protein_final"], reverse=True)


def _have_source(results: list[dict]) -> bool:
    return any("r2_protein_old" in r or "r2_protein_new" in r for r in results)


def _fmt(v, w=6):
    if v is None:
        return f"{'—':>{w}}"
    return f"{v:>{w}.3f}"


def print_table(results: list[dict]) -> None:
    if not results:
        print("No results.")
        return
    has_src = _have_source(results)
    if has_src:
        header = (
            f"\n{'run':<55}  {'n':>4}  {'R²(p)':>7}  {'R²(m)':>7}  "
            f"{'R²(p|old)':>9}  {'R²(m|old)':>9}  {'n_old':>5}  "
            f"{'R²(p|new)':>9}  {'R²(m|new)':>9}  {'n_new':>5}"
        )
        print(header)
        print("-" * len(header))
        for r in results:
            name = re.sub(r"^\d{8}_\d{6}_", "", r["run"])
            print(
                f"{name:<55}  {r['n']:>4}  "
                f"{r['r2_protein_final']:>7.3f}  {r['r2_mrna_max']:>7.3f}  "
                f"{_fmt(r.get('r2_protein_old'), 9)}  {_fmt(r.get('r2_mrna_old'), 9)}  "
                f"{(r.get('n_old') if r.get('n_old') is not None else '—'):>5}  "
                f"{_fmt(r.get('r2_protein_new'), 9)}  {_fmt(r.get('r2_mrna_new'), 9)}  "
                f"{(r.get('n_new') if r.get('n_new') is not None else '—'):>5}"
            )
    else:
        print(f"\n{'run':<55}  {'n':>4}  {'R²(protein final)':>18}  {'R²(mRNA max)':>13}")
        print("-" * 97)
        for r in results:
            name = re.sub(r"^\d{8}_\d{6}_", "", r["run"])
            print(f"{name:<55}  {r['n']:>4}  {r['r2_protein_final']:>18.4f}  {r['r2_mrna_max']:>13.4f}")

    best = results[0]
    best_name = re.sub(r"^\d{8}_\d{6}_", "", best["run"])
    extra = ""
    if "r2_protein_old" in best or "r2_protein_new" in best:
        parts = []
        if "r2_protein_old" in best:
            parts.append(f"old={best['r2_protein_old']:.3f}")
        if "r2_protein_new" in best:
            parts.append(f"new={best['r2_protein_new']:.3f}")
        extra = "  [protein " + ", ".join(parts) + "]"
    print(f"\nBest: {best_name}  "
          f"R²(protein)={best['r2_protein_final']:.4f}  R²(mRNA)={best['r2_mrna_max']:.4f}{extra}")


def save_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BASE_FIELDS)
    if _have_source(results):
        for k in SOURCE_FIELDS:
            if any(k in r for r in results):
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            row["run"] = re.sub(r"^\d{8}_\d{6}_", "", r["run"])
            for k in fieldnames:
                if k.startswith("r2_") and row[k] != "":
                    row[k] = f"{row[k]:.6f}"
            writer.writerow(row)
    print(f"Saved CSV → {path}")


def plot_comparison(results: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [re.sub(r"^\d{8}_\d{6}_", "", r["run"]) for r in results]
    r2_prot = [r["r2_protein_final"] for r in results]
    r2_mrna = [r["r2_mrna_max"]      for r in results]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.9), 5))
    w = 0.35
    bars1 = ax.bar(x - w / 2, r2_prot, w, label="R²(protein final)", color="steelblue")
    bars2 = ax.bar(x + w / 2, r2_mrna, w, label="R²(mRNA max)",       color="darkorange")

    for bar, val in zip(list(bars1) + list(bars2), r2_prot + r2_mrna):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("R²  (higher is better)")
    ax.set_title("Endpoint R² comparison")
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Study folder, e.g. experiments/txtl_supervisor_combined_sweep")
    parser.add_argument("--recompute", action="store_true", help="Ignore cache and recompute all")
    parser.add_argument("--csv",       type=str, default=None, help="Save summary CSV to this path")
    parser.add_argument("--plot",      action="store_true",    help="Save a grouped bar chart")
    parser.add_argument("--plot-out",  type=str, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    results = load_or_compute(root, recompute=args.recompute)
    print_table(results)

    if args.csv:
        save_csv(results, Path(args.csv))
    if args.plot:
        out = Path(args.plot_out) if args.plot_out else root / "r2_comparison.png"
        plot_comparison(results, out)
