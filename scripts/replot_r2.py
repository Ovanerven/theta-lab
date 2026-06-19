"""Regenerate endpoint_r2.png, endpoint_r2_by_source.png, and r2_cache.csv
for every finished run in a study folder, using the gate-matched eval.

Usage:
    python scripts/replot_r2.py experiments_final/old_only
    python scripts/replot_r2.py experiments_final/old_only --plot-traj
    python scripts/replot_r2.py experiments_final/experiments/good_m9_runs --recompute

Notes:
  - Reads each run's saved config.yaml to apply subtract_channel_min correctly
    via the fix in last-layer-ode/metrics/endpoint_r2.py.
  - Skips runs missing model.pt or split.npz.
  - With --plot-traj, also regenerates pred_vs_true_*.png and theta plots
    via plot_predictions (slower; ~30s/run extra).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "last-layer-ode"))

import numpy as np
import torch
from metrics.endpoint_r2 import (
    collect_endpoints, r2, save_r2_cache, plot_endpoints,
    plot_endpoints_by_source,
)
from plot_diagnostics import (
    rebuild_model_from_experiment, device_auto, load_yaml,
    plot_predictions,
)


def _find_runs(root: Path, allow_partial: bool = False) -> list[Path]:
    """Find runs with at least a config + split + some checkpoint.

    By default requires model.pt (best-val). With allow_partial=True, also
    accepts runs that only have model_last.pt yet — useful for peeking at
    runs that are still training. rebuild_model_from_experiment falls back
    to model_last.pt automatically when model.pt is missing.
    """
    def has_ckpt(d: Path) -> bool:
        if allow_partial:
            return (d / "model.pt").exists() or (d / "model_last.pt").exists()
        return (d / "model.pt").exists()

    if (root / "config.yaml").exists() and has_ckpt(root):
        return [root]
    return sorted(
        d for d in root.iterdir()
        if d.is_dir()
        and (d / "config.yaml").exists()
        and has_ckpt(d)
        and (d / "split.npz").exists()
    )


def replot_one(exp_dir: Path, device, recompute: bool, plot_traj: bool):
    print(f"  {exp_dir.name}")
    try:
        raw = collect_endpoints(exp_dir, device, split="test",
                                protein_sp="pm", mrna_sp="mm")
    except Exception as e:
        print(f"    SKIP collect_endpoints: {e}")
        return

    r2p = r2(raw["true_protein_final"], raw["pred_protein_final"])
    r2m = r2(raw["true_mrna_max"],      raw["pred_mrna_max"])
    print(f"    R²(p)={r2p:.4f}  R²(m)={r2m:.4f}")

    # 1) cache CSV (per-source columns persisted)
    save_r2_cache(exp_dir, raw, r2p, r2m)

    # 2) overall scatter
    plot_endpoints(
        [raw], protein_sp="pm", mrna_sp="mm", split="test",
        out_path=exp_dir / "endpoint_r2.png",
    )

    # 3) by-source scatter (no-op if dataset has no source labels)
    plot_endpoints_by_source(
        raw, protein_sp="pm", mrna_sp="mm", split="test",
        out_path=exp_dir / "endpoint_r2_by_source.png",
    )

    # 4) trajectory plots (optional, slower)
    if plot_traj:
        try:
            cfg = load_yaml(exp_dir / "config.yaml")
            model, ds, obs_names, _, lift_info = rebuild_model_from_experiment(exp_dir, device)
            scaffold_names = (lift_info or {}).get("scaffold_state_names")
            state_names = scaffold_names if scaffold_names else (list(obs_names) if obs_names is not None else [])
            out_dir = exp_dir / "plots"
            out_dir.mkdir(exist_ok=True)
            plot_predictions(
                model, ds, state_names=state_names, out_dir=out_dir,
                n_samples=int(cfg.get("plot_samples", 5)),
                device=device, exp_dir=exp_dir,
                u_transform=str(cfg.get("u_transform", "none")),
                param_names=None, lift_info=lift_info,
                extra_new_samples=int(cfg.get("plot_extra_new_samples", 5)),
            )
            print(f"    trajectory plots → {out_dir}")
        except Exception as e:
            print(f"    SKIP plot_predictions: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", help="Study folder OR single run dir")
    p.add_argument("--recompute", action="store_true",
                   help="Ignored — this script always recomputes.")
    p.add_argument("--plot-traj", action="store_true",
                   help="Also regenerate pred_vs_true and theta plots (slower).")
    p.add_argument("--allow-partial", action="store_true",
                   help="Include runs that only have model_last.pt (e.g. still training).")
    args = p.parse_args()

    root = Path(args.root)
    runs = _find_runs(root, allow_partial=args.allow_partial)
    if not runs:
        print(f"No finished runs in {root}")
        sys.exit(1)

    device = device_auto()
    print(f"Replotting {len(runs)} run(s) in {root} (device={device})")
    for exp_dir in runs:
        replot_one(exp_dir, device, args.recompute, args.plot_traj)
    print("Done.")
