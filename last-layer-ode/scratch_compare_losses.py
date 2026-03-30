"""Throwaway: compare log1p-MSE vs MSE loss by NRMSE on test splits.

Folders:
  log1p-MSE: experiments/mof_synthesis_baselines   (study: mof_synthesis_baselines)
  MSE:       experiments/mof_synthesis_mse          (study: mof_synthesis_mse)

Usage (from repo root):
    python last-layer-ode/scratch_compare_losses.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_diagnostics import rebuild_model_from_experiment, device_auto, _test_subset

LOG1P_ROOT = Path("experiments/mof_synthesis_baselines")
MSE_ROOT   = Path("experiments/mof_synthesis_mse")


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    rng = float(np.max(y_true) - np.min(y_true))
    return rmse / max(rng, 1e-8)


def find_runs(root: Path) -> dict[str, Path]:
    """Find completed runs recursively. Returns {scaffold: latest_run_dir}."""
    by_scaffold: dict[str, list[Path]] = defaultdict(list)
    for cfg_path in sorted(root.rglob("config.yaml")):
        exp_dir = cfg_path.parent
        if not (exp_dir / "model.pt").exists():
            continue
        if not (exp_dir / "split.npz").exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        scaffold = cfg.get("scaffold")
        if scaffold:
            by_scaffold[scaffold].append(exp_dir)
    # pick latest run per scaffold (sorted by run dir name = timestamp)
    return {sc: sorted(dirs)[-1] for sc, dirs in by_scaffold.items()}


def eval_run(exp_dir: Path, device: torch.device) -> dict[str, float]:
    """Returns {species: median_nrmse} for the test split."""
    model, ds, obs_names, _ = rebuild_model_from_experiment(exp_dir, device)
    test_subset = _test_subset(ds, exp_dir)
    dt = torch.tensor(ds.dt.astype(np.float32)).to(device)
    obs_idx = torch.arange(len(obs_names), device=device)

    model.eval()
    species_vals: dict[str, list[float]] = {s: [] for s in obs_names}

    with torch.no_grad():
        for i in range(len(test_subset)):
            y0, u_seq, y_seq = test_subset[i]
            pred, _, _ = model(
                y0.unsqueeze(0).to(device),
                u_seq.unsqueeze(0).to(device),
                dt.unsqueeze(0),
                obs_idx,
                y_seq=None,
                teacher_forcing=False,
            )
            y_np = y_seq.cpu().numpy()
            p_np = pred[0].cpu().numpy()
            for j, sp in enumerate(obs_names):
                species_vals[sp].append(nrmse(y_np[:, j], p_np[:, j]))

    return {sp: float(np.median(vals)) for sp, vals in species_vals.items()}


def main():
    device = device_auto()

    log1p_runs = find_runs(LOG1P_ROOT)
    mse_runs   = find_runs(MSE_ROOT)

    scaffolds = sorted(set(log1p_runs) | set(mse_runs),
                       key=lambda s: int(''.join(filter(str.isdigit, s)) or 0))

    print(f"\nComparing log1p-MSE vs MSE loss  —  median NRMSE on test split")
    print(f"{'scaffold':<22}  {'log1p-MSE':>10}  {'MSE':>10}  winner")
    print("-" * 58)

    for scaffold in scaffolds:
        if scaffold not in log1p_runs or scaffold not in mse_runs:
            print(f"{scaffold:<22}  (missing from one side, skipping)")
            continue

        print(f"  evaluating {scaffold}...", end="\r")
        log1p_scores = eval_run(log1p_runs[scaffold], device)
        mse_scores   = eval_run(mse_runs[scaffold], device)

        log1p_mean = float(np.mean(list(log1p_scores.values())))
        mse_mean   = float(np.mean(list(mse_scores.values())))
        winner = "log1p-MSE" if log1p_mean < mse_mean else "MSE      "

        print(f"{scaffold:<22}  {log1p_mean:>10.4f}  {mse_mean:>10.4f}  {winner}")
        for sp in sorted(set(log1p_scores) | set(mse_scores)):
            l = log1p_scores.get(sp, float("nan"))
            m = mse_scores.get(sp, float("nan"))
            print(f"  {sp:<20}  {l:>10.4f}  {m:>10.4f}")

    print()


if __name__ == "__main__":
    main()
