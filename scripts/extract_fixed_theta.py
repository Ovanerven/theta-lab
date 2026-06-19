#!/usr/bin/env python3
"""
Extract the fitted global theta from stage-1 ode_fixed_theta runs and print a
ready-to-paste `pin_theta:` line for the stage-2 NODE_corr sweeps.

Stage 1 (FINAL_coarse_node_corr_stage1_fixedtheta.yaml) fits one global theta to
all cell-free trajectories with no neural correction. This reads each run's
model.pt, undoes the log_gamma bounding to recover the physical rate constants,
and emits the dict to paste into:
    FINAL_coarse_node_corr_stage2_pintheta_h{400,600}.yaml

Usage:
    python scripts/extract_fixed_theta.py experiments/FINAL_coarse_node_corr_stage1_fixedtheta
    python scripts/extract_fixed_theta.py <study_dir> --pick s0   # force a seed instead of best-val
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# M4 (txtl_model4_three_state) rate-constant names, in theta order.
# TXTLModel4_ThreeStateScaffold: theta_dim=5.
PARAM_NAMES = ["v_TX", "v_TL", "k_M", "k_mat", "k_degp"]


def log_gamma(raw, lo, hi):
    # inverse of the bounding used in OdeFixedTheta / NeuralOdeCorrection:
    # theta = lo * exp(log(hi/lo) * sigmoid(raw))
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(raw))


def best_val_loss(run_dir: Path):
    """Best (min) validation loss from logs/loss_curves.npz, or None if absent."""
    npz = run_dir / "logs" / "loss_curves.npz"
    if not npz.exists():
        return None
    try:
        d = np.load(npz)
    except Exception:
        return None
    for key in ("val_loss", "val", "val_losses", "valid_loss"):
        if key in d and len(d[key]):
            return float(np.nanmin(d[key]))
    return None


def physical_theta(model_pt: Path):
    ckpt = torch.load(model_pt, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    raw = sd["raw_theta"].float()
    lo = sd["theta_lo_vec"].float()
    hi = sd["theta_hi_vec"].float()
    return log_gamma(raw, lo, hi)


def fmt_dict(theta):
    items = ", ".join(f"{i}: {v:.4g}" for i, v in enumerate(theta.tolist()))
    return "{" + items + "}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study_dir", help="stage-1 study dir (contains <timestamp>_<exp_name>/ run dirs)")
    ap.add_argument("--pick", default=None,
                    help="force a seed tag (e.g. s0) instead of the best-val run")
    args = ap.parse_args()

    study = Path(args.study_dir)
    runs = sorted(p.parent for p in study.glob("*/model.pt"))
    if not runs:
        sys.exit(f"no */model.pt found under {study}")

    rows = []  # (run_dir, theta, val)
    for rd in runs:
        theta = physical_theta(rd / "model.pt")
        rows.append((rd, theta, best_val_loss(rd)))

    name_w = max(len(n) for n in PARAM_NAMES)
    print(f"\nStage-1 global theta — {study.name}\n")
    for rd, theta, val in rows:
        vals = theta.tolist()
        if len(vals) != len(PARAM_NAMES):
            print(f"  WARNING: {len(vals)} theta dims but {len(PARAM_NAMES)} names "
                  f"in PARAM_NAMES — labels below may be wrong (values are fine).")
        vstr = f"  (best val {val:.5g})" if val is not None else ""
        print(f"{rd.name}{vstr}")
        for i, v in enumerate(vals):
            n = PARAM_NAMES[i] if i < len(PARAM_NAMES) else f"theta[{i}]"
            print(f"    [{i}] {n:<{name_w}}  {v:.4g}")
        print(f"    pin_theta: {fmt_dict(theta)}\n")

    stack = torch.stack([t for _, t, _ in rows])
    geomean = torch.exp(torch.log(stack).mean(dim=0))

    # Recommendation: best-val seed if val curves exist, else geometric mean.
    pick = None
    if args.pick:
        pick = next((r for r in rows if args.pick in r[0].name), None)
        if pick is None:
            sys.exit(f"--pick {args.pick} matched no run")
    elif any(v is not None for _, _, v in rows):
        pick = min((r for r in rows if r[2] is not None), key=lambda r: r[2])

    print("=" * 60)
    print("PASTE INTO stage-2 specs (same line in both h400 and h600):\n")
    if pick is not None:
        print(f"  # from {pick[0].name} (best val)")
        print(f"  pin_theta: {fmt_dict(pick[1])}")
    print(f"\n  # or geometric mean across {len(rows)} seeds:")
    print(f"  pin_theta: {fmt_dict(geomean)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
