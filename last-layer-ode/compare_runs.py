"""
compare_runs.py  —  print and optionally plot a comparison table of test losses
across all runs under a given experiment root directory.

Usage:
    python last-layer-ode/compare_runs.py experiments/mof_arch_ablation
    python last-layer-ode/compare_runs.py experiments/mof_arch_ablation --plot
    python last-layer-ode/compare_runs.py experiments/mof_arch_ablation --csv results/mof_arch_ablation.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def collect_runs(root: Path) -> list[dict]:
    rows = []
    for npz in sorted(root.rglob("logs/loss_curves.npz")):
        exp_dir = npz.parent.parent
        d = np.load(str(npz), allow_pickle=True)

        test_loss = float(d["test_loss"]) if "test_loss" in d and d["test_loss"].ndim == 0 else None
        if test_loss is None:
            continue

        species_losses = d["test_species_losses"].tolist() if "test_species_losses" in d else []

        # read species names from config if available
        cfg_path = exp_dir / "config.yaml"
        species_names = []
        if cfg_path.exists():
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text())
            ds_path = Path(cfg.get("dataset_path", ""))
            if ds_path.exists():
                ds = np.load(str(ds_path), allow_pickle=True)
                if "obs_names" in ds:
                    species_names = ds["obs_names"].astype(str).tolist()

        val_losses = d["val_losses"] if "val_losses" in d else np.array([])
        best_val = float(val_losses.min()) if len(val_losses) > 0 else None

        rows.append({
            "run": exp_dir.name,
            "exp_dir": str(exp_dir),
            "test_loss": test_loss,
            "best_val": best_val,
            "species_losses": species_losses,
            "species_names": species_names,
        })

    rows.sort(key=lambda r: r["test_loss"])
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No completed runs found.")
        return

    # collect all species names
    all_species: list[str] = []
    for r in rows:
        for s in r["species_names"]:
            if s not in all_species:
                all_species.append(s)

    col_w = 52
    sp_w  = 10

    header = f"{'run':<{col_w}}  {'test_loss':>10}  {'best_val':>10}"
    for s in all_species:
        header += f"  {s[:sp_w]:>{sp_w}}"
    print(header)
    print("-" * len(header))

    for r in rows:
        line = f"{r['run']:<{col_w}}  {r['test_loss']:>10.6f}  "
        line += f"{r['best_val']:>10.6f}" if r["best_val"] is not None else f"{'N/A':>10}"
        sp_map = dict(zip(r["species_names"], r["species_losses"]))
        for s in all_species:
            val = sp_map.get(s)
            line += f"  {val:>{sp_w}.6f}" if val is not None else f"  {'N/A':>{sp_w}}"
        print(line)


def save_csv(rows: list[dict], path: Path) -> None:
    import csv
    all_species: list[str] = []
    for r in rows:
        for s in r["species_names"]:
            if s not in all_species:
                all_species.append(s)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "test_loss", "best_val"] + all_species)
        for r in rows:
            import re
            name = re.sub(r"^\d{8}_\d{6}_", "", r["run"])
            sp_map = dict(zip(r["species_names"], r["species_losses"]))
            writer.writerow(
                [name, r["test_loss"], r["best_val"] or ""]
                + [sp_map.get(s, "") for s in all_species]
            )
    print(f"Saved CSV to {path}")


def plot_comparison(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    names = [r["run"] for r in rows]
    losses = [r["test_loss"] for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.8), 5))
    bars = ax.bar(range(len(names)), losses)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Test loss (MSE)")
    ax.set_title("Test loss comparison")

    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="Experiment root dir, e.g. experiments/mof_arch_ablation")
    parser.add_argument("--plot", action="store_true", help="Save a bar chart")
    parser.add_argument("--plot-out", type=str, default=None, help="Path for the bar chart (default: <root>/comparison.png)")
    parser.add_argument("--csv", type=str, default=None, help="Save a CSV summary to this path")
    args = parser.parse_args()

    root = Path(args.root)
    rows = collect_runs(root)
    print_table(rows)

    if args.csv:
        save_csv(rows, Path(args.csv))

    if args.plot:
        out = Path(args.plot_out) if args.plot_out else root / "comparison.png"
        plot_comparison(rows, out)
