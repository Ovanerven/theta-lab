"""
oracle_fit_cfps.py

Oracle per-step theta fit for the CFPS (cell-free TXTL) dataset, across
heterogeneous scaffolds (M3..M9 of the final ladder).

Unlike per_step_theta_fit*.py (which assume dataset state == scaffold state and
only work for M5), this harness replicates train.py's data handling exactly:

  * partial-observability LIFT: dataset is a fixed 7-channel layout
    ['R','O','m','mm','p','pm','DNA'] with mm@3 (Broccoli mRNA) and pm@5
    (mCherry protein) the only measured channels. Each scaffold places those at
    its own obs_state_idx and copies resource ICs (R,O,DNA) by name into y0
    (train._lift_to_scaffold_state).
  * bolus jump rebuilt from scaffold.control_state_map (not dataset indices).
  * per-sample dt from dt_per_sample (NOT np.diff(t_obs), which is a merged grid).
  * per-sample length masking (samples padded to K=324; lengths 215..324).
  * subtract_channel_min on the observed cols (matches config).

Only the HONEST ROLLOUT is computed: at each step we start from the model's own
previous prediction, GD-optimise theta to hit the next observation, and advance
from the prediction. Errors compound. This is the "max achievable fit" of the
scaffold's structure with a freely-chosen theta(t). (One-step teacher forcing is
invalid here: latent channels are zero in y_seq, so resetting to them each step
kills the dynamics.)

Reported per scaffold: pooled trajectory R^2 and NRMSE on mm (mRNA) and pm
(protein), over all valid (sample, timestep) points, plus endpoint R^2.

Usage:
    python analysis/oracle_fit_cfps.py \
        --dataset datasets/cell-free/txtl_native_real_only_coarsenold.npz \
        --scaffolds txtl_model3_two_state,txtl_model4_three_state,\
txtl_resource_and_maturation_dna,txtl_model7_bg_fixed,\
txtl_model8_bg_fixed,txtl_model9_event_dark \
        --gd-steps 250 --out results/oracle_fit_cfps
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scaffolds import SCAFFOLDS
from jumps import make_u_to_y_jump
import train as T  # reuse _lift_to_scaffold_state, _apply_channel_min_gate


# ── canonical ladder name map (for pretty labels) ───────────────────────────────
LADDER_LABEL = {
    "txtl_model3_two_state": "M3",
    "txtl_model4_three_state": "M4",
    "txtl_resource_and_maturation_dna": "M5",
    "txtl_model7_bg_fixed": "M7",
    "txtl_model8_bg_fixed": "M8",
    "txtl_model9_event_dark": "M9",
}
DEFAULT_SCAFFOLDS = list(LADDER_LABEL.keys())
DATASET_OBS_IDX = [3, 5]   # mm (mRNA), pm (protein) columns in the npz layout


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_gamma(x, lo, hi):
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x))


STATE_CAP = 1.0e6   # legit states are < ~1e4; this only catches free-theta blow-ups


def rk4_batch(rhs, y, dt, theta, n_sub=3):
    """y:(B,P), dt:(B,) or (B,1), theta:(B,theta_dim).

    State is clamped to [0, STATE_CAP] after EVERY substep. With bounded theta,
    an unstable scaffold + stiff RK4 can otherwise drive a few samples to inf
    over a long rollout, producing NaN gradients that poison the whole batch.
    Clamping never touches real trajectories (all < ~1e4) — it only bounds the
    pathological blow-ups so the joint optimiser sees a large-but-finite loss and
    learns to avoid them."""
    if dt.ndim == 1:
        dt = dt.unsqueeze(1)
    h = dt / float(n_sub)
    for _ in range(n_sub):
        k1 = rhs(y, theta)
        k2 = rhs(y + 0.5 * h * k1, theta)
        k3 = rhs(y + 0.5 * h * k2, theta)
        k4 = rhs(y + h * k3, theta)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # clamp() alone does NOT remove NaN (only inf); some scaffold RHS terms
        # (0/0, inf*0 in rate laws) emit NaN directly. nan_to_num first, then bound.
        y = torch.clamp(torch.nan_to_num(y, nan=0.0, posinf=STATE_CAP, neginf=0.0),
                        0.0, STATE_CAP)
    return y


def r2_pooled(pred, true, valid):
    """pred,true,valid: (N,K) for one channel. Pooled R^2 over valid points."""
    p = pred[valid]
    t = true[valid]
    if t.size == 0:
        return float("nan")
    ss_res = float(np.sum((p - t) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def nrmse_pooled(pred, true, valid):
    p = pred[valid]
    t = true[valid]
    if t.size == 0:
        return float("nan")
    rng = float(t.max() - t.min())
    if rng < 1e-10:
        return float("nan")
    return float(np.sqrt(np.mean((p - t) ** 2)) / rng)


def build_jump(scaffold, ds_control_names, ds_control_indices, ds_obs_indices, P, device):
    """Identity jump if scaffold matches dataset (M5); else from control_state_map."""
    if scaffold.obs_state_idx is None and scaffold.control_state_map is None:
        return make_u_to_y_jump(ds_control_indices, ds_obs_indices, device=device)
    U = len(ds_control_names)
    J = torch.zeros((U, P), dtype=torch.float32, device=device)
    for j, name in enumerate(ds_control_names):
        target = scaffold.control_state_map.get(str(name).strip())
        if target is None:
            continue
        targets = list(target) if isinstance(target, (list, tuple)) else [target]
        for t in targets:
            J[j, int(t)] = 1.0
    return J


def _fit_greedy(sc, y0_f, y_seq_f, u_seq, dt, jump, lo_t, hi_t,
                obs_idx_t, valid, args, label, device, N, K):
    """Greedy per-step rollout oracle: at each step optimise theta_k to hit the
    NEXT observation, starting from own previous prediction. Warm-started across
    steps. Can diverge when the scaffold cannot represent the protein dynamics."""
    theta_dim = lo_t.shape[0]
    pred_obs = torch.zeros((N, K, obs_idx_t.shape[0]), device=device)
    y_cur = y0_f.clone()
    raw = torch.zeros(N, theta_dim, device=device)
    for k in range(K):
        y_after = (y_cur + u_seq[:, k] @ jump).detach()
        # clamp_min: channel-min gate sends PADDED targets negative; log1p(<-1)=NaN
        # and NaN*0(mask) still poisons the sum. Valid targets are already >=0.
        target = y_seq_f[:, k].index_select(1, obs_idx_t).clamp_min(0.0)
        step_valid = valid[:, k].unsqueeze(1).float()
        raw_k = raw.clone().requires_grad_(True)
        opt = torch.optim.Adam([raw_k], lr=args.lr)
        for _ in range(args.gd_steps):
            opt.zero_grad()
            theta_k = log_gamma(raw_k, lo_t, hi_t)
            y_hat = rk4_batch(sc, y_after, dt[:, k], theta_k, args.n_substeps)
            diff = (torch.log1p(y_hat.index_select(1, obs_idx_t))
                    - torch.log1p(target)) * step_valid
            (diff.pow(2).sum() / step_valid.sum().clamp_min(1.0)).backward()
            opt.step()
        with torch.no_grad():
            raw = raw_k.detach()
            theta_k = log_gamma(raw, lo_t, hi_t)
            y_hat = rk4_batch(sc, y_after, dt[:, k], theta_k, args.n_substeps)
            pred_obs[:, k] = y_hat.index_select(1, obs_idx_t)
            keep = valid[:, k].unsqueeze(1).float()
            y_cur = keep * y_hat + (1.0 - keep) * y_cur
        if (k + 1) % 50 == 0 or k == K - 1:
            print(f"    [{label}] greedy step {k+1}/{K}", flush=True)
    return pred_obs


def _fit_joint(sc, y0_f, y_seq_f, u_seq, dt, jump, lo_t, hi_t,
               obs_idx_t, valid_f, args, label, device, N, K):
    """Joint theta(t) trajectory oracle: optimise the FULL (N,K,theta_dim) theta
    against the whole-rollout observed-channel loss (backprop through all K
    steps). This is the true achievable ceiling and is stable — exploding the
    rollout only increases the loss being minimised. Chunked over samples."""
    theta_dim = lo_t.shape[0]
    pred_obs = torch.zeros((N, K, obs_idx_t.shape[0]), device=device)
    chunk = args.chunk
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        y0_c = y0_f[s:e]; yseq_c = y_seq_f[s:e]; u_c = u_seq[s:e]
        dt_c = dt[s:e]; vf_c = valid_f[s:e]
        B = e - s
        raw = torch.zeros(B, K, theta_dim, device=device, requires_grad=True)
        opt = torch.optim.Adam([raw], lr=args.lr)
        tgt = yseq_c.index_select(2, obs_idx_t).clamp_min(0.0)   # (B,K,n_obs); see greedy note
        for it in range(args.gd_steps):
            opt.zero_grad()
            theta = log_gamma(raw, lo_t, hi_t)            # (B,K,theta_dim)
            y = y0_c
            preds = []
            for k in range(K):
                y = rk4_batch(sc, y + u_c[:, k] @ jump, dt_c[:, k],
                              theta[:, k], args.n_substeps)
                preds.append(y.index_select(1, obs_idx_t))
            pred = torch.stack(preds, dim=1)              # (B,K,n_obs)
            diff = (torch.log1p(pred) - torch.log1p(tgt)) * vf_c.unsqueeze(-1)
            loss = diff.pow(2).sum() / vf_c.sum().clamp_min(1.0)
            loss.backward()
            # sanitise: a blown-up sample can still leave inf/nan grads — zero them
            # and clip so one pathological trajectory can't poison Adam's state.
            if raw.grad is not None:
                torch.nan_to_num_(raw.grad, nan=0.0, posinf=0.0, neginf=0.0)
                torch.nn.utils.clip_grad_norm_([raw], max_norm=1e3)
            opt.step()
            if (it + 1) % max(1, args.gd_steps // 4) == 0:
                print(f"    [{label}] joint chunk {s}-{e} it {it+1}/{args.gd_steps} "
                      f"loss={float(loss):.4f}", flush=True)
        with torch.no_grad():
            theta = log_gamma(raw, lo_t, hi_t)
            y = y0_c
            for k in range(K):
                y = rk4_batch(sc, y + u_c[:, k] @ jump, dt_c[:, k],
                              theta[:, k], args.n_substeps)
                pred_obs[s:e, k] = y.index_select(1, obs_idx_t)
    return pred_obs


def fit_scaffold(scaffold_name, ds, args, device):
    sc = SCAFFOLDS[scaffold_name]
    P = sc.P
    theta_dim = sc.theta_dim
    label = LADDER_LABEL.get(scaffold_name, scaffold_name)

    N = ds["y0"].shape[0]
    if args.n_samples and args.n_samples > 0:
        N = min(N, args.n_samples)
    sl = slice(0, N)

    y0_np = ds["y0"][sl].astype(np.float32)
    y_seq_np = ds["y_seq"][sl].astype(np.float32)
    u_seq_np = ds["u_seq"][sl].astype(np.float32)
    dt_np = ds["dt_per_sample"][sl].astype(np.float32)
    lengths = ds["lengths"][sl].astype(np.int64)
    K = y_seq_np.shape[1]

    y0 = torch.from_numpy(y0_np).to(device)
    y_seq = torch.from_numpy(y_seq_np).to(device)
    u_seq = torch.from_numpy(u_seq_np).to(device)
    dt = torch.from_numpy(dt_np).to(device)
    L = torch.from_numpy(lengths).to(device)

    # subtract_channel_min on observed cols [3,5] (matches config), with length mask
    y0, y_seq = T._apply_channel_min_gate(y0, y_seq, cols=DATASET_OBS_IDX, lengths=L)

    # lift to scaffold state space
    if sc.obs_state_idx is None and sc.control_state_map is None:
        # M5: identity layout (dataset == scaffold)
        y0_f, y_seq_f = y0, y_seq
        scaf_obs_idx = DATASET_OBS_IDX
    else:
        y0_f, y_seq_f = T._lift_to_scaffold_state(
            y0, y_seq,
            dataset_obs_idx=DATASET_OBS_IDX,
            scaffold_obs_idx=sc.obs_state_idx,
            scaffold_P=P,
            dataset_state_names=[str(n) for n in ds["obs_names"]],
            scaffold_state_names=list(sc.state_names),
        )
        scaf_obs_idx = sc.obs_state_idx

    jump = build_jump(
        sc, list(ds["control_names"]),
        torch.from_numpy(ds["control_indices"].astype(np.int64)),
        torch.from_numpy(ds["obs_indices"].astype(np.int64)),
        P, device,
    )

    # theta bounds
    if sc.theta_lo_vec is not None and sc.theta_hi_vec is not None:
        lo_t = torch.tensor(sc.theta_lo_vec, dtype=torch.float32, device=device)
        hi_t = torch.tensor(sc.theta_hi_vec, dtype=torch.float32, device=device)
    else:
        lo_t = torch.full((theta_dim,), args.theta_lo, device=device)
        hi_t = torch.full((theta_dim,), args.theta_hi, device=device)

    obs_idx_t = torch.tensor(scaf_obs_idx, dtype=torch.long, device=device)

    # valid[n,k] = step k is a real (non-padded) observation
    ar = torch.arange(K, device=device).unsqueeze(0)
    valid = ar < L.unsqueeze(1)                     # (N,K)
    valid_f = valid.float()

    if args.mode == "joint":
        pred_obs = _fit_joint(sc, y0_f, y_seq_f, u_seq, dt, jump, lo_t, hi_t,
                              obs_idx_t, valid_f, args, label, device, N, K)
    else:
        pred_obs = _fit_greedy(sc, y0_f, y_seq_f, u_seq, dt, jump, lo_t, hi_t,
                               obs_idx_t, valid, args, label, device, N, K)

    # ── metrics on observed channels (mm=mRNA, pm=protein) ──────────────────────
    valid_np = valid.cpu().numpy()
    pred_np = pred_obs.cpu().numpy()
    true_np = y_seq_f.index_select(2, obs_idx_t).cpu().numpy()  # (N,K,n_obs)

    ch_names = ["mRNA", "protein"]
    rows = {}
    for j, cn in enumerate(ch_names):
        r2 = r2_pooled(pred_np[:, :, j], true_np[:, :, j], valid_np)
        nr = nrmse_pooled(pred_np[:, :, j], true_np[:, :, j], valid_np)
        rows[cn] = (r2, nr)

    # endpoint R^2: last valid step per sample
    last_idx = (lengths - 1)
    ep_pred = pred_np[np.arange(N), last_idx]   # (N,n_obs)
    ep_true = true_np[np.arange(N), last_idx]
    ep = {}
    for j, cn in enumerate(ch_names):
        t = ep_true[:, j]; p = ep_pred[:, j]
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        ep[cn] = (1.0 - float(np.sum((p - t) ** 2)) / ss_tot) if ss_tot > 1e-12 else float("nan")

    return dict(
        scaffold=scaffold_name, label=label, P=P, theta_dim=theta_dim, N=N,
        traj=rows, endpoint=ep,
        pred=pred_np, true=true_np, valid=valid_np,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/cell-free/txtl_native_real_only_coarsenold.npz")
    ap.add_argument("--scaffolds", default=",".join(DEFAULT_SCAFFOLDS))
    ap.add_argument("--gd-steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-substeps", type=int, default=3)
    ap.add_argument("--theta-lo", type=float, default=1e-6)
    ap.add_argument("--theta-hi", type=float, default=2.0)
    ap.add_argument("--mode", default="joint", choices=["joint", "greedy"],
                    help="joint = optimise full theta(t) vs whole-rollout loss "
                         "(stable ceiling); greedy = per-step rollout (can diverge).")
    ap.add_argument("--chunk", type=int, default=256, help="sample chunk for joint mode")
    ap.add_argument("--n-samples", type=int, default=-1, help="-1 = all")
    ap.add_argument("--out", default="results/oracle_fit_cfps")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else device_auto()
    ds_path = args.dataset
    if not Path(ds_path).is_absolute():
        ds_path = str(Path(__file__).resolve().parent.parent.parent / ds_path)
    ds = np.load(ds_path, allow_pickle=True)
    scaffolds = [s.strip() for s in args.scaffolds.split(",") if s.strip()]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Dataset: {ds_path}  (N={ds['y0'].shape[0]})")
    print(f"gd_steps={args.gd_steps} lr={args.lr} n_substeps={args.n_substeps}\n")

    # Incremental CSV: write header once, append each scaffold's row as it
    # finishes, flush+fsync. A later OOM (SIGABRT, uncatchable) then cannot wipe
    # already-computed scaffolds. Per-scaffold predictions also saved immediately.
    csv_path = out_dir / "oracle_fit_summary.csv"
    header = ["scaffold", "label", "P", "theta_dim", "N",
              "trajR2_mRNA", "trajR2_protein", "nrmse_mRNA", "nrmse_protein",
              "endpointR2_mRNA", "endpointR2_protein"]
    write_header = not csv_path.exists()
    csv_f = open(csv_path, "a", newline="")
    w = csv.writer(csv_f)
    if write_header:
        w.writerow(header)
        csv_f.flush()

    base_chunk = args.chunk
    for sn in scaffolds:
        if sn not in SCAFFOLDS:
            print(f"[warn] unknown scaffold {sn} — skipping")
            continue
        sc = SCAFFOLDS[sn]
        # auto-shrink chunk for heavy scaffolds: the joint autograd graph scales
        # with chunk*P*K; cap so P=12 models don't OOM (SIGABRT).
        args.chunk = max(64, min(base_chunk, int(base_chunk * 7 / max(sc.P, 1))))
        print(f">> {LADDER_LABEL.get(sn, sn)}  {sn}  (P={sc.P}, theta_dim={sc.theta_dim}, chunk={args.chunk})", flush=True)
        res = fit_scaffold(sn, ds, args, device)
        tr = res["traj"]; ep = res["endpoint"]
        print(f"   trajR2  mRNA={tr['mRNA'][0]:+.3f}  protein={tr['protein'][0]:+.3f}"
              f"   endpointR2 mRNA={ep['mRNA']:+.3f} protein={ep['protein']:+.3f}\n", flush=True)
        w.writerow([res["scaffold"], res["label"], res["P"], res["theta_dim"], res["N"],
                    f"{tr['mRNA'][0]:.4f}", f"{tr['protein'][0]:.4f}",
                    f"{tr['mRNA'][1]:.4f}", f"{tr['protein'][1]:.4f}",
                    f"{ep['mRNA']:.4f}", f"{ep['protein']:.4f}"])
        csv_f.flush()
        os.fsync(csv_f.fileno())
        np.savez_compressed(out_dir / f"pred_{res['label']}.npz",
                            pred=res["pred"], true=res["true"], valid=res["valid"])
    csv_f.close()
    print(f"Saved summary -> {csv_path}")


if __name__ == "__main__":
    main()
