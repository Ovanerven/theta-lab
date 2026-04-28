"""Deep diagnosis of training failure on real_ivtt data.

Tests several hypotheses for why the model fails to learn trajectories:

  H1. Mean-trajectory collapse (preds ≈ dataset mean)
      -> compute corr(pred_i, mean_pred), corr(pred_i, mean_true)

  H2. log_mse attractor at geomean
      -> compare model val loss vs "predict geometric mean" baseline.
         if similar, log_mse is choosing scale-invariant minima.

  H3. Open-loop rollout divergence (model OK with TF, fails autoregressively)
      -> compute val loss with teacher_forcing=True every K and compare
         against open-loop loss

  H4. Theta saturation (sigmoid in log_gamma is stuck)
      -> distribution of head pre-sigmoid logits per parameter

  H5. Compounding error (early prediction error propagates)
      -> per-timestep MSE: where in the trajectory does error blow up?

  H6. Per-step pred variance collapse
      -> std across batch at each timestep — when does variance die?

Usage:
    python last-layer-ode/diagnose_deep.py <run_dir> [--n 64]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scaffolds import SCAFFOLDS
from models import MODELS
from jumps import make_u_to_y_jump


def device_auto():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def log_mse(pred, true):
    return np.mean((np.log1p(np.maximum(pred, 0)) - np.log1p(np.maximum(true, 0))) ** 2)


def mse(pred, true):
    return np.mean((pred - true) ** 2)


def r2(true, pred):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--dataset", default="datasets/real_ivtt_full.npz")
    ap.add_argument("--n", type=int, default=64, help="trajectories to use")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    dev = device_auto()
    print(f"device: {dev}\nrun: {run_dir.name}")

    # ── load model ────────────────────────────────────────────────────────
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    ckpt = torch.load(run_dir / "model.pt", map_location=dev, weights_only=False)
    state = ckpt["state_dict"]
    print(f"cfg: hidden={cfg.get('hidden')}, lr={cfg.get('lr')}, "
          f"y_transform={cfg.get('y_transform')}, loss_type={cfg.get('loss_type')}, "
          f"lambda_endpoint={cfg.get('lambda_endpoint', 0)}")

    d = np.load(args.dataset, allow_pickle=True)
    y0_all = torch.from_numpy(d["y0"].astype(np.float32))
    u_all = torch.from_numpy(d["u_seq"].astype(np.float32))
    y_all = torch.from_numpy(d["y_seq"].astype(np.float32))
    t_obs = d["t_obs"].astype(np.float32)
    dt = torch.from_numpy(np.diff(t_obs).astype(np.float32))
    lengths_np = d["lengths"].astype(np.int64)
    lengths = torch.from_numpy(lengths_np)
    ci = torch.from_numpy(d["control_indices"].astype(np.int64))
    oi = torch.from_numpy(d["obs_indices"].astype(np.int64))

    sc = SCAFFOLDS[cfg["scaffold"]]
    U = u_all.shape[-1]
    P = sc.P
    K_data = u_all.shape[1]

    gru_u_cols = [j for j in range(U) if int(ci[j]) >= P] if cfg.get("exclude_ode_cols_from_gru", False) else None
    gru_y_cols = list(cfg.get("obs_idx", [])) if cfg.get("gru_y_obs_only", False) else None
    jump = make_u_to_y_jump(ci, oi, device=dev)

    model = MODELS[cfg.get("model_class", "ode_rnn_txtl")](
        U=U, rhs=sc, u_to_y_jump=jump,
        hidden=cfg.get("hidden", 128),
        lift_dim=cfg.get("lift_dim", 32),
        num_layers=cfg.get("num_layers", 1),
        dropout=0.0,
        ff_mult=cfg.get("ff_mult", 2),
        theta_lo=cfg.get("theta_lo", 1e-6),
        theta_hi=cfg.get("theta_hi", 1.0),
        n_substeps=1, use_basal=False,
        context_len=64, tf_group_size=32, ar_gap=4,
        theta_bounded=True,
        d_state=16, expand=2, d_conv=4,
        forget_bias_init=None, legacy_forget_bias_bug=False,
        gru_u_cols=gru_u_cols,
        gru_y_cols=gru_y_cols,
        head_bias_init=cfg.get("head_bias_init", 0.0),
        head_weight_gain=cfg.get("head_weight_gain", 1.0),
    ).to(dev)
    model.load_state_dict(state, strict=False)
    model.eval()

    # ── pick val trajectories ─────────────────────────────────────────────
    split = run_dir / "split.npz"
    if split.exists():
        idx = np.load(split)["val_idx"][: args.n]
    else:
        idx = np.arange(args.n)
    N = len(idx)

    sel = torch.from_numpy(idx)
    y0_b = y0_all[sel].to(dev)
    u_b = u_all[sel].to(dev)
    y_b = y_all[sel].to(dev)
    K = u_b.shape[1]
    dt_b = dt[None, :K].expand(N, -1).to(dev)
    obs_idx_t = torch.tensor(cfg.get("obs_idx", [3, 5]), device=dev)
    L_min = int(lengths[sel].min().item())

    # ── rollout: open-loop, TF every 50, full TF ─────────────────────────
    def run(tf, every):
        with torch.no_grad():
            pred, theta, _ = model(
                y0_b, u_b, dt_b, obs_idx_t,
                y_seq=y_b if tf else None,
                teacher_forcing=tf, tf_every=every,
                u_transform=cfg.get("u_transform", "cumsum"),
                y_transform=cfg.get("y_transform", "none"),
            )
        return pred.cpu().numpy(), theta.cpu().numpy()

    print("\nRunning rollouts: open-loop, TF50, TF1 ...")
    pred_ol, theta_ol = run(False, 50)
    pred_tf50, _ = run(True, 50)
    pred_tf1, _ = run(True, 1)
    y_np = y_b.cpu().numpy()

    # =======================================================================
    print("\n" + "=" * 80)
    print("H1. MEAN-TRAJECTORY COLLAPSE — pred trajectories all look like the mean?")
    print("=" * 80)
    for sp_idx, name in [(3, "mm"), (5, "pm")]:
        true = y_np[:, :L_min, sp_idx]
        pred = pred_ol[:, :L_min, sp_idx]
        true_mean = true.mean(axis=0)
        pred_mean = pred.mean(axis=0)
        # corr between each trajectory and the mean trajectory
        corrs_pred = [np.corrcoef(pred[i], pred_mean)[0, 1] for i in range(N)]
        corrs_true = [np.corrcoef(true[i], true_mean)[0, 1] for i in range(N)]
        print(f"  {name}: mean(corr(pred_i, mean_pred)) = {np.nanmean(corrs_pred):.3f}  "
              f"mean(corr(true_i, mean_true)) = {np.nanmean(corrs_true):.3f}")
        print(f"      → if pred-corr >> true-corr, pred trajectories are clones of the mean")

    # =======================================================================
    print("\n" + "=" * 80)
    print("H2. log_mse ATTRACTOR — does 'predict geometric mean' beat the model?")
    print("=" * 80)
    for sp_idx, name in [(3, "mm"), (5, "pm")]:
        true = y_np[:, :L_min, sp_idx]
        pred = pred_ol[:, :L_min, sp_idx]
        # geomean per-time across trajectories
        geomean_t = np.exp(np.mean(np.log1p(np.maximum(true, 0)), axis=0)) - 1
        # baseline: every traj predicted as the geomean curve
        baseline = np.broadcast_to(geomean_t, true.shape)
        m_logmse = log_mse(pred, true)
        b_logmse = log_mse(baseline, true)
        m_mse = mse(pred, true)
        b_mse = mse(baseline, true)
        print(f"  {name}: model log_mse={m_logmse:.4f} vs geomean-baseline={b_logmse:.4f}   "
              f"(model {'BETTER' if m_logmse < b_logmse else 'WORSE'})")
        print(f"      model     MSE = {m_mse:>10.2f}  vs baseline = {b_mse:>10.2f}   "
              f"(model {'BETTER' if m_mse < b_mse else 'WORSE'})")

    # =======================================================================
    print("\n" + "=" * 80)
    print("H3. OPEN-LOOP DIVERGENCE — does TF rescue the model?")
    print("=" * 80)
    print(f"  open-loop  log_mse(mm,pm) = {log_mse(pred_ol[:,:L_min,3], y_np[:,:L_min,3]):.4f}, "
          f"{log_mse(pred_ol[:,:L_min,5], y_np[:,:L_min,5]):.4f}")
    print(f"  TF every50 log_mse(mm,pm) = {log_mse(pred_tf50[:,:L_min,3], y_np[:,:L_min,3]):.4f}, "
          f"{log_mse(pred_tf50[:,:L_min,5], y_np[:,:L_min,5]):.4f}")
    print(f"  TF every1  log_mse(mm,pm) = {log_mse(pred_tf1[:,:L_min,3], y_np[:,:L_min,3]):.4f}, "
          f"{log_mse(pred_tf1[:,:L_min,5], y_np[:,:L_min,5]):.4f}")
    print("      → if TF1 ≪ TF50 ≪ open-loop, model can't propagate without help (compounding)")

    # =======================================================================
    print("\n" + "=" * 80)
    print("H4. THETA HEAD SATURATION — are pre-sigmoid logits at the boundaries?")
    print("=" * 80)
    # logits before log_gamma. Theta = lo * exp(log(hi/lo) * sigmoid(x))
    # We want: sigmoid(x) distribution. log(theta/lo) / log(hi/lo) = sigmoid(x).
    theta_lo = cfg.get("theta_lo", 1e-6)
    theta_hi = cfg.get("theta_hi", 1.0)
    log_ratio = np.log(theta_hi / theta_lo)
    theta_names = ["lam", "lam_O", "VTXmax", "kdm", "VTLmax", "kmt", "kmatm"]
    sig = np.clip(np.log(np.maximum(theta_ol, 1e-30) / theta_lo) / log_ratio, 0, 1)
    for j, nm in enumerate(theta_names):
        s = sig[:, :L_min, j]
        frac_low = (s < 0.05).mean()
        frac_high = (s > 0.95).mean()
        print(f"  {nm:>7}: sigmoid mean={s.mean():.3f} std={s.std():.3f}  "
              f"saturated_low={frac_low:.2%}  saturated_high={frac_high:.2%}")
    print("      → if many saturated, log_gamma blocks gradient flow")

    # =======================================================================
    print("\n" + "=" * 80)
    print("H5. ERROR PROPAGATION — at what timestep does open-loop blow up?")
    print("=" * 80)
    for sp_idx, name in [(3, "mm"), (5, "pm")]:
        true = y_np[:, :L_min, sp_idx]
        pred = pred_ol[:, :L_min, sp_idx]
        # per-step log_mse averaged across batch
        per_t = ((np.log1p(np.maximum(pred, 0)) - np.log1p(np.maximum(true, 0))) ** 2).mean(axis=0)
        bins = [(0, L_min // 8), (L_min // 8, L_min // 4), (L_min // 4, L_min // 2),
                (L_min // 2, 3 * L_min // 4), (3 * L_min // 4, L_min)]
        labels = ["t<L/8", "L/8-L/4", "L/4-L/2", "L/2-3L/4", "3L/4-L"]
        print(f"  {name}: " + "  ".join([f"{lab}={per_t[a:b].mean():.3f}"
                                         for lab, (a, b) in zip(labels, bins)]))

    # =======================================================================
    print("\n" + "=" * 80)
    print("H6. PRED-VARIANCE COLLAPSE — when does cross-traj variance die?")
    print("=" * 80)
    for sp_idx, name in [(3, "mm"), (5, "pm")]:
        true = y_np[:, :L_min, sp_idx]
        pred = pred_ol[:, :L_min, sp_idx]
        true_std = true.std(axis=0)
        pred_std = pred.std(axis=0)
        ratio = pred_std / np.maximum(true_std, 1e-6)
        bins = [(0, L_min // 8), (L_min // 8, L_min // 4), (L_min // 4, L_min // 2),
                (L_min // 2, 3 * L_min // 4), (3 * L_min // 4, L_min)]
        labels = ["t<L/8", "L/8-L/4", "L/4-L/2", "L/2-3L/4", "3L/4-L"]
        print(f"  {name}:  pred/true cross-traj std ratio")
        print(f"        " + "  ".join([f"{lab}={ratio[a:b].mean():.2f}"
                                       for lab, (a, b) in zip(labels, bins)]))

    # =======================================================================
    print("\n" + "=" * 80)
    print("RANK-CORRELATION — does the model preserve ORDER even if scale is wrong?")
    print("=" * 80)
    from scipy.stats import spearmanr, pearsonr
    for sp_idx, name in [(3, "mm-peak"), (5, "pm-final")]:
        true_eps = []
        pred_eps = []
        for i in range(N):
            L = int(lengths[sel[i]])
            if name == "mm-peak":
                true_eps.append(y_np[i, :L, sp_idx].max())
                pred_eps.append(pred_ol[i, :L, sp_idx].max())
            else:
                true_eps.append(y_np[i, L - 1, sp_idx])
                pred_eps.append(pred_ol[i, L - 1, sp_idx])
        true_eps = np.array(true_eps)
        pred_eps = np.array(pred_eps)
        sp = spearmanr(true_eps, pred_eps).statistic
        pe = pearsonr(true_eps, pred_eps).statistic
        print(f"  {name}:  Pearson r = {pe:.3f}    Spearman r = {sp:.3f}")
    print("      → if Spearman > 0 but Pearson ~ 0, model gets ranking right; just scale wrong")
    print("        (=> log_mse is letting the model 'compress' outputs into a narrow band)")


if __name__ == "__main__":
    main()
