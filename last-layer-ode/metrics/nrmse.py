"""Shared NRMSE computation and caching for compare_runs and plot_nrmse.

Cache file: <run_dir>/nrmse_cache.csv  (one file per run, travels with the run)
One row per species. Stores per-run aggregated stats (median, mean, std, q25, q75).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_diagnostics import rebuild_model_from_experiment, device_auto, _test_subset, _filter_model_kwargs
from scaffolds import SCAFFOLDS

CACHE_NAME = "nrmse_cache.csv"
CACHE_FIELDS = ["run", "scaffold", "P", "species", "n_samples",
                "median", "mean", "std", "q25", "q75"]


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Black-box rollouts (e.g. neural_ode_mlp) can diverge to ±inf over long
    # horizons. Replace non-finite preds with a large finite sentinel scaled to
    # y_true's magnitude so a diverged model scores "huge but finite" rather
    # than NaN — keeps the CSV/aggregation usable.
    if not np.all(np.isfinite(y_pred)):
        scale = float(np.max(np.abs(y_true))) if y_true.size else 1.0
        sentinel = 1e6 * max(scale, 1.0)
        y_pred = np.nan_to_num(y_pred, nan=sentinel, posinf=sentinel, neginf=-sentinel)
    with np.errstate(over="ignore", invalid="ignore"):
        diff = y_pred.astype(np.float64) - y_true.astype(np.float64)
        rmse = float(np.sqrt(np.mean(diff * diff)))
    rng = float(np.max(y_true) - np.min(y_true))
    return rmse / max(rng, 1e-8)


def _find_exp_dirs(exp_root: Path) -> list[Path]:
    return sorted(
        d for d in exp_root.iterdir()
        if d.is_dir()
        and (d / "config.yaml").exists()
        and (d / "model.pt").exists()
        and (d / "split.npz").exists()
    )


def _compute_run(exp_dir: Path, device: torch.device, no_split: bool = False,
                 dataset_override: "str | None" = None) -> list[dict]:
    import yaml

    cfg = yaml.safe_load((exp_dir / "config.yaml").read_text())
    if dataset_override is not None:
        cfg["dataset_path"] = dataset_override
    scaffold = cfg.get("scaffold", exp_dir.name)
    P = SCAFFOLDS[scaffold].P if scaffold in SCAFFOLDS else 0

    model, ds, obs_names, _, lift_info = rebuild_model_from_experiment(exp_dir, device)
    test_subset = ds if no_split else _test_subset(ds, exp_dir)

    model.eval()
    # dt now read per-sample from item[3] (the dataset's __getitem__).

    # When lifting, restrict species accounting to the actually-observed pair
    # (mRNA, protein). Labels come from the scaffold's state_names at those
    # positions; ground truth from the dataset at its obs positions.
    if lift_info:
        scaffold_obs = list(lift_info["scaffold_obs_idx"])
        dataset_obs = list(lift_info["dataset_obs_idx"])
        species_labels = [obs_names[i] if i < len(obs_names) else f"y{i}" for i in scaffold_obs]
    else:
        scaffold_obs = list(range(len(obs_names)))
        dataset_obs = list(range(len(obs_names)))
        species_labels = list(obs_names)
    species_vals: dict[str, list[float]] = {s: [] for s in species_labels}

    base_kwargs = {
        "y_seq": None,
        "teacher_forcing": False,
        "u_transform": str(cfg.get("u_transform", "none")),
        "y_transform": str(cfg.get("y_transform", "none")),
    }

    with torch.no_grad():
        for i in range(len(test_subset)):
            item = test_subset[i]
            y0, u_seq, y_seq, dt_i = item[0], item[1], item[2], item[3]
            y0_b = y0.unsqueeze(0).to(device)
            u_b = u_seq.unsqueeze(0).to(device)
            y_seq_b = y_seq.unsqueeze(0).to(device)
            dt_b = dt_i.unsqueeze(0).to(device)
            if lift_info:
                from plot_diagnostics import _maybe_lift
                y0_b, _ = _maybe_lift(y0_b, y_seq_b, lift_info)
                obs_idx_t = torch.tensor(scaffold_obs, device=device, dtype=torch.long)
            else:
                obs_idx_t = torch.arange(len(obs_names), device=device)
            pred, _, _ = model(
                y0_b, u_b, dt_b, obs_idx_t,
                **_filter_model_kwargs(model, base_kwargs),
            )
            y_np = y_seq.cpu().numpy()             # dataset layout
            p_np = pred[0].cpu().numpy()           # scaffold layout
            for k, sp in enumerate(species_labels):
                d_col = dataset_obs[k] if k < len(dataset_obs) else k
                s_col = scaffold_obs[k] if k < len(scaffold_obs) else k
                species_vals[sp].append(nrmse(y_np[:, d_col], p_np[:, s_col]))

    rows = []
    for sp, vals in species_vals.items():
        v = np.array(vals)
        rows.append({
            "run": exp_dir.name,
            "scaffold": scaffold,
            "P": P,
            "species": sp,
            "n_samples": len(v),
            "median": float(np.median(v)),
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "q25": float(np.quantile(v, 0.25)),
            "q75": float(np.quantile(v, 0.75)),
        })
    return rows


def _save_cache(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CACHE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _load_cache(path: Path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["P"] = int(r["P"])
        r["n_samples"] = int(r["n_samples"])
        for col in ("median", "mean", "std", "q25", "q75"):
            r[col] = float(r[col])
    return rows


def load_or_compute(exp_root: Path, recompute: bool = False,
                    no_split: bool = False,
                    dataset_override: "str | None" = None) -> list[dict]:
    """Return NRMSE rows for all runs under exp_root.

    Each run's results are cached in <run_dir>/nrmse_cache.csv so they travel
    with the run if folders are rearranged. Only runs without a cache are
    recomputed (unless recompute=True forces all).

    Each row: {run, scaffold, P, species, n_samples, median, mean, std, q25, q75}
    """
    device = device_auto()
    exp_dirs = _find_exp_dirs(exp_root)
    if not exp_dirs:
        print(f"No completed runs found in {exp_root}")
        return []

    all_rows: list[dict] = []

    for exp_dir in exp_dirs:
        cache_path = exp_dir / CACHE_NAME

        if cache_path.exists() and not recompute and dataset_override is None:
            rows = _load_cache(cache_path)
            cache_has_nan = any(
                not np.isfinite(r[c])
                for r in rows for c in ("median", "mean", "std", "q25", "q75")
            )
            if cache_has_nan:
                print(f"  {exp_dir.name}  (stale NaN cache — recomputing)")
            else:
                print(f"  {exp_dir.name}  (cached)")
                all_rows.extend(rows)
                continue

        try:
            rows = _compute_run(exp_dir, device, no_split=no_split,
                                dataset_override=dataset_override)
        except Exception as e:
            print(f"  skip {exp_dir.name}: {e}")
            continue

        _save_cache(rows, cache_path)
        mean_nrmse = float(np.mean([r["median"] for r in rows]))
        print(f"  {exp_dir.name}  mean-median-NRMSE={mean_nrmse:.4f}  (saved cache)")
        all_rows.extend(rows)

    return all_rows
