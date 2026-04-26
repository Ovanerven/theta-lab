"""
Diagnostic for the materials-scenario training collapse.

Loads a trained checkpoint and asks:
  1. Does the model produce DIFFERENT predictions for DIFFERENT experiments?
  2. Or has it collapsed to a single 'mean trajectory'?

If the per-experiment predictions are nearly identical regardless of the reagent
mix, the GRU is ignoring its input — i.e. it has converged to the degenerate
"predict the mean" fixed point. This is what the prediction-vs-truth plots from
real_ivtt_test/20260425_000808_stepwise_dna look like.

Run from project root:
  python last-layer-ode/diagnose_collapse.py \
      --ckpt experiments/real_ivtt_test/<RUN>/model_last.pt \
      --dataset datasets/real_ivtt_full.npz \
      --n 8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scaffolds import SCAFFOLDS
from models import MODELS
from jumps import make_u_to_y_jump


def device_auto():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="datasets/real_ivtt_full.npz")
    ap.add_argument("--n", type=int, default=8, help="how many trajectories to compare")
    ap.add_argument("--scaffold", default="txtl_resource_and_maturation_dna")
    ap.add_argument("--model-class", default="ode_rnn_txtl")
    args = ap.parse_args()

    dev = device_auto()
    print(f"device: {dev}")

    ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
    state = ckpt["state_dict"]
    cfg = ckpt.get("cfg", {})
    print(f"loaded ckpt: hidden={cfg.get('hidden', '?')} num_layers={cfg.get('num_layers', '?')} "
          f"u_transform={cfg.get('u_transform', '?')} "
          f"exclude_ode={cfg.get('exclude_ode_cols_from_gru', '?')}")

    d = np.load(args.dataset, allow_pickle=True)
    y0 = torch.from_numpy(d["y0"].astype(np.float32))
    u_seq = torch.from_numpy(d["u_seq"].astype(np.float32))
    y_seq = torch.from_numpy(d["y_seq"].astype(np.float32))
    t_obs = d["t_obs"].astype(np.float32)
    dt = torch.from_numpy(np.diff(t_obs).astype(np.float32))
    lengths = torch.from_numpy(d["lengths"].astype(np.int64))
    ci = torch.from_numpy(d["control_indices"].astype(np.int64))
    oi = torch.from_numpy(d["obs_indices"].astype(np.int64))

    sc = SCAFFOLDS[args.scaffold]
    U = u_seq.shape[-1]
    P = sc.P

    gru_u_cols = None
    if cfg.get("exclude_ode_cols_from_gru", False):
        gru_u_cols = [j for j in range(U) if int(ci[j]) >= P]

    jump = make_u_to_y_jump(ci, oi, device=dev)

    model = MODELS[args.model_class](
        U=U, rhs=sc, u_to_y_jump=jump,
        hidden=cfg.get("hidden", 256),
        lift_dim=cfg.get("lift_dim", 32),
        num_layers=cfg.get("num_layers", 2),
        dropout=0.0,
        ff_mult=cfg.get("ff_mult", 2),
        theta_lo=cfg.get("theta_lo", 1e-6),
        theta_hi=cfg.get("theta_hi", 1.0),
        n_substeps=1,
        use_basal=False,
        context_len=64, tf_group_size=32, ar_gap=4,
        theta_bounded=True,
        d_state=16, expand=2, d_conv=4,
        forget_bias_init=None, legacy_forget_bias_bug=False,
        gru_u_cols=gru_u_cols,
        head_bias_init=cfg.get("head_bias_init", 0.0),
        head_weight_gain=cfg.get("head_weight_gain", 1.0),
    ).to(dev)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Pick first n trajectories
    N = min(args.n, y0.shape[0])
    sel = torch.arange(N)
    y0_b = y0[sel].to(dev)
    u_b = u_seq[sel].to(dev)
    y_b = y_seq[sel].to(dev)
    K = u_b.shape[1]
    dt_b = dt[None, :K].expand(N, -1).to(dev)
    obs_idx = torch.tensor([3, 5], device=dev)

    with torch.no_grad():
        pred, theta, _ = model(
            y0_b, u_b, dt_b, obs_idx,
            y_seq=None, teacher_forcing=False,
            tf_every=50, tbptt_chunk=0,
            u_transform=cfg.get("u_transform", "cumsum"),
        )

    pred_np = pred.cpu().numpy()       # (N, K, P)
    theta_np = theta.cpu().numpy()     # (N, K, theta_dim)
    y_np = y_b.cpu().numpy()           # (N, K, P)

    # Per-trajectory pm and mm peaks/finals
    print()
    print("=" * 80)
    print("Per-trajectory pred vs true — does the model differentiate experiments?")
    print("=" * 80)
    print(f"{'i':>2} | {'true mm peak':>12} {'pred mm peak':>12} | {'true pm fin':>11} {'pred pm fin':>11}")
    for i in range(N):
        L = int(lengths[sel[i]])
        true_mm_peak = y_np[i, :L, 3].max()
        pred_mm_peak = pred_np[i, :L, 3].max()
        true_pm_fin = y_np[i, L - 1, 5]
        pred_pm_fin = pred_np[i, L - 1, 5]
        print(f"{i:>2} | {true_mm_peak:>12.2f} {pred_mm_peak:>12.2f} | "
              f"{true_pm_fin:>11.2f} {pred_pm_fin:>11.2f}")

    # Spread of predictions across trajectories: if model collapsed, std should be tiny
    print()
    print("=" * 80)
    print("Cross-trajectory variance of predictions vs truth (higher = more differentiation)")
    print("=" * 80)
    L = min(int(lengths[sel].min()), pred_np.shape[1])
    for sp_idx, sp_name in [(3, "mm"), (5, "pm")]:
        true_std = y_np[:, :L, sp_idx].std(axis=0).mean()  # mean over time of std-across-traj
        pred_std = pred_np[:, :L, sp_idx].std(axis=0).mean()
        ratio = pred_std / max(true_std, 1e-6)
        print(f"  {sp_name}: true cross-traj std = {true_std:>8.3f}  "
              f"pred std = {pred_std:>8.3f}  ratio = {ratio:.3f}  "
              f"{'<< COLLAPSED' if ratio < 0.2 else '<< low' if ratio < 0.5 else 'OK'}")

    # Theta variance across trajectories: collapsed model has constant theta
    print()
    print("=" * 80)
    print("Cross-trajectory variance of THETA (B,K,theta_dim) — constant theta = collapsed")
    print("=" * 80)
    theta_names = ["lam", "lam_O", "VTXmax", "kdm", "VTLmax", "kmt", "kmatm"]
    for j, nm in enumerate(theta_names):
        # mean over time and per-trajectory
        per_traj_mean = theta_np[:, :L, j].mean(axis=1)  # (N,)
        cv = per_traj_mean.std() / max(abs(per_traj_mean.mean()), 1e-12)
        print(f"  theta[{j}]={nm:>7}: per-traj-mean min={per_traj_mean.min():.3g} "
              f"max={per_traj_mean.max():.3g} CV={cv:.3f}")


if __name__ == "__main__":
    main()
