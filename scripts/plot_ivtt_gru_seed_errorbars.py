from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


METRIC_COLUMNS = ("r2_protein_final", "r2_mrna_max")


def _read_run_config(cfg_path: Path) -> dict[str, str]:
    text = cfg_path.read_text(errors="ignore")
    out: dict[str, str] = {}
    for key in ("study", "exp_name", "seed", "model_class"):
        m = re.search(rf"^{key}:\s*(.+?)\s*$", text, re.M)
        if m:
            out[key] = m.group(1).strip()
    return out


def _read_metrics(cache_path: Path) -> dict[str, float]:
    with cache_path.open(newline="") as f:
        row = next(csv.DictReader(f), None)
    if not row:
        raise ValueError(f"No metrics row found in {cache_path}")
    metrics: dict[str, float] = {}
    for key in METRIC_COLUMNS:
        metrics[key] = float(row[key])
    return metrics


def collect_runs(root: Path, seed_filter: int | None = None) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for cfg_path in sorted(root.glob("experiments/**/config.yaml")):
        cfg = _read_run_config(cfg_path)
        if cfg.get("model_class") != "ode_rnn":
            continue
        if cfg.get("study") not in {"ivtt_gru", "ivtt_gru_drop5000"}:
            continue
        cache = cfg_path.parent / "r2_cache.csv"
        if not cache.exists():
            continue
        seed = int(cfg["seed"])
        if seed_filter is not None and seed != seed_filter:
            continue
        metrics = _read_metrics(cache)
        runs.append(
            {
                "run": cfg_path.parent.name,
                "study": cfg.get("study", ""),
                "exp_name": cfg.get("exp_name", ""),
                "study_exp_name": f"{cfg.get('study', '')}_{cfg.get('exp_name', '')}".strip("_"),
                "seed": seed,
                **metrics,
            }
        )
    return runs


def summarize_by_seed(runs: list[dict[str, object]]) -> dict[int, dict[str, float]]:
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        seed = int(run["seed"])
        for key in METRIC_COLUMNS:
            grouped[seed][key].append(float(run[key]))

    summary: dict[int, dict[str, float]] = {}
    for seed, metrics in grouped.items():
        summary[seed] = {}
        for key, vals in metrics.items():
            arr = np.asarray(vals, dtype=float)
            summary[seed][f"{key}_mean"] = float(arr.mean())
            summary[seed][f"{key}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            summary[seed][f"{key}_n"] = float(arr.size)
    return summary


def save_runs_csv(runs: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "study", "exp_name", "study_exp_name", "seed", "r2_protein_final", "r2_mrna_max"])
        for run in sorted(runs, key=lambda r: str(r["run"])):
            writer.writerow([
                run["run"],
                run["study"],
                run["exp_name"],
                run["study_exp_name"],
                int(run["seed"]),
                f"{float(run['r2_protein_final']):.6f}",
                f"{float(run['r2_mrna_max']):.6f}",
            ])


def plot_same_seed_scatter(runs: list[dict[str, object]], out_path: Path, seed: int | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
        }
    )

    x = np.asarray([float(r["r2_mrna_max"]) for r in runs], dtype=float)
    y = np.asarray([float(r["r2_protein_final"]) for r in runs], dtype=float)
    labels = [str(r.get("study_exp_name", r["run"])) for r in runs]

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    fig.patch.set_facecolor("white")

    ax.scatter(x, y, s=42, color="#1f77b4", alpha=0.95)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=7)

    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    pad = max(0.02, 0.08 * (hi - lo if hi > lo else 1.0))
    ax.set_xlim(max(0.0, lo - pad), min(1.0, hi + pad))
    ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
    diag_lo, diag_hi = ax.get_xlim()[0], ax.get_xlim()[1]
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], linestyle="--", linewidth=1.0, color="#7f7f7f", alpha=0.8)

    ax.set_xlabel("mRNA R² (max)")
    ax.set_ylabel("Protein R² (final)")
    title_seed = f"seed={seed}" if seed is not None else "all seeds"
    ax.set_title(f"Repeated runs: {title_seed}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_seed_summary(summary: dict[int, dict[str, float]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
        }
    )

    seeds = sorted(summary)
    x = np.arange(len(seeds), dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True)
    fig.patch.set_facecolor("white")
    titles = {
        "r2_protein_final": "Protein final R² by seed",
        "r2_mrna_max": "mRNA max R² by seed",
    }

    for ax, metric in zip(axes, METRIC_COLUMNS):
        means = [summary[s][f"{metric}_mean"] for s in seeds]
        errs = [summary[s][f"{metric}_std"] for s in seeds]

        ax.errorbar(
            x,
            means,
            yerr=errs,
            fmt="o",
            capsize=4,
            linewidth=1.6,
            markersize=6,
            color="#1f77b4",
            ecolor="#1f77b4",
        )
        for xi, seed, mean, err in zip(x, seeds, means, errs):
            ax.text(xi, mean + (err if err > 0 else 0.008), f"{mean:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(titles[metric])
        ax.set_ylabel("R²")
        ax.set_ylim(0.0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.8)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(s) for s in seeds])
    axes[-1].set_xlabel("Seed")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_different_seed_variation(runs: list[dict[str, object]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
        }
    )

    seeds = sorted({int(r["seed"]) for r in runs})
    x_base = np.arange(len(seeds), dtype=float)
    seed_to_x = {s: i for i, s in enumerate(seeds)}

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True)
    fig.patch.set_facecolor("white")

    titles = {
        "r2_protein_final": "Protein final R² across seeds",
        "r2_mrna_max": "mRNA max R² across seeds",
    }

    for ax, metric in zip(axes, METRIC_COLUMNS):
        y_by_seed: dict[int, list[float]] = defaultdict(list)
        for r in runs:
            s = int(r["seed"])
            y = float(r[metric])
            y_by_seed[s].append(y)
            jitter = float(rng.uniform(-0.08, 0.08))
            ax.scatter(seed_to_x[s] + jitter, y, s=24, alpha=0.85, color="#1f77b4")

        means = []
        stds = []
        for s in seeds:
            arr = np.asarray(y_by_seed[s], dtype=float)
            means.append(float(arr.mean()))
            stds.append(float(arr.std(ddof=1)) if arr.size > 1 else 0.0)

        ax.errorbar(
            x_base,
            means,
            yerr=stds,
            fmt="o",
            color="#d62728",
            ecolor="#d62728",
            capsize=4,
            linewidth=1.5,
            markersize=5,
        )
        ax.set_title(titles[metric])
        ax.set_ylabel("R²")
        ax.set_ylim(0.0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.8)

    axes[-1].set_xticks(x_base)
    axes[-1].set_xticklabels([str(s) for s in seeds])
    axes[-1].set_xlabel("Seed")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_summary_csv(summary: dict[int, dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "r2_protein_final_mean", "r2_protein_final_std", "r2_protein_final_n", "r2_mrna_max_mean", "r2_mrna_max_std", "r2_mrna_max_n"])
        for seed in sorted(summary):
            row = summary[seed]
            writer.writerow([
                seed,
                f"{row['r2_protein_final_mean']:.6f}",
                f"{row['r2_protein_final_std']:.6f}",
                int(row['r2_protein_final_n']),
                f"{row['r2_mrna_max_mean']:.6f}",
                f"{row['r2_mrna_max_std']:.6f}",
                int(row['r2_mrna_max_n']),
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot endpoint R² with error bars across ivtt_gru seeds.")
    parser.add_argument("--root", type=Path, default=Path("."), help="theta-lab workspace root")
    parser.add_argument("--out", type=Path, default=Path("results/ivtt_gru_seed_errorbars.pdf"), help="Output plot path")
    parser.add_argument("--csv", type=Path, default=Path("results/ivtt_gru_seed_errorbars.csv"), help="Output summary CSV path")
    parser.add_argument("--seed", type=int, default=None, help="Optional: only include this seed (e.g., 42)")
    parser.add_argument("--mode", choices=["seed_errorbars", "same_seed_scatter", "different_seed_variation"], default="seed_errorbars",
                        help="Plot mode: aggregate error bars by seed, or per-run scatter at (typically) one seed")
    args = parser.parse_args()

    runs = collect_runs(args.root, seed_filter=args.seed)
    if not runs:
        if args.seed is None:
            raise SystemExit("No matching ivtt_gru / ivtt_gru_drop5000 runs with r2_cache.csv found.")
        raise SystemExit(f"No matching ivtt_gru / ivtt_gru_drop5000 runs with r2_cache.csv found for seed={args.seed}.")

    if args.mode == "same_seed_scatter":
        save_runs_csv(runs, args.csv)
        plot_same_seed_scatter(runs, args.out, args.seed)
        print(f"Saved plot to {args.out}")
        print(f"Saved per-run CSV to {args.csv}")
        return

    if args.mode == "different_seed_variation":
        save_runs_csv(runs, args.csv)
        plot_different_seed_variation(runs, args.out)
        print(f"Saved plot to {args.out}")
        print(f"Saved per-run CSV to {args.csv}")
        return

    summary = summarize_by_seed(runs)
    save_summary_csv(summary, args.csv)
    plot_seed_summary(summary, args.out)
    print(f"Saved plot to {args.out}")
    print(f"Saved summary CSV to {args.csv}")


if __name__ == "__main__":
    main()