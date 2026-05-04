"""Run Bob's supervisor checkpoint on the real_ivtt dataset and report R².

Goal: disambiguate whether the "pm collapses to mean" failure is data or
training. If Bob's own weights yield strong R² on a chosen test split, our
training is at fault; if they collapse here too, his published figures came
from different data.

Usage:
    python run_supervisor_inference.py \
        --ckpt ../GRU_O_mat_6005.pt \
        --dataset ../datasets/real_ivtt_raw_no_outliers.npz \
        --split-seed 12 --n-test 104 \
        --out-dir ../experiments/supervisor_ckpt_eval/sp12_6005

Reusable: vary --split-seed / --n-test / --ckpt to sweep test splits and
checkpoints. R² rows append to <out-dir>/r2_cache.csv across runs sharing the
same out-dir parent.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.gru_supervisor import GRU_model_latent_decay_simple_fb_matO


def stratified_split(y_seq, lengths, n_test, n_val, seed, n_bins=5, targets=(3, 5)):
    """Mirror train_R.py's stratified split: bin by endpoint values of given
    full-state targets, then evenly draw test/val/train across bins."""
    N = y_seq.shape[0]
    end_t = np.clip(lengths.astype(np.int64) - 1, 0, y_seq.shape[1] - 1)
    rows = np.arange(N)
    end_vals = np.stack([y_seq[rows, end_t, t] for t in targets], axis=1)  # (N, T)

    rng = np.random.default_rng(seed)
    bin_id = np.zeros(N, dtype=np.int64)
    for j in range(end_vals.shape[1]):
        v = end_vals[:, j]
        try:
            qs = np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1])
            bj = np.digitize(v, qs)
        except Exception:
            bj = np.zeros(N, dtype=np.int64)
        bin_id = bin_id * n_bins + bj

    order = np.argsort(bin_id, kind="stable")
    perm = []
    by_bin = {}
    for i in order:
        by_bin.setdefault(int(bin_id[i]), []).append(int(i))
    for b in by_bin.values():
        rng.shuffle(b)
    cursors = {b: 0 for b in by_bin}
    keys = list(by_bin.keys())
    rng.shuffle(keys)

    pulled = []
    while len(pulled) < N:
        for b in keys:
            if cursors[b] < len(by_bin[b]):
                pulled.append(by_bin[b][cursors[b]])
                cursors[b] += 1
            if len(pulled) >= N:
                break
    pulled = np.array(pulled, dtype=np.int64)
    test_idx = pulled[:n_test]
    val_idx = pulled[n_test:n_test + n_val]
    train_idx = pulled[n_test + n_val:]
    return train_idx, val_idx, test_idx


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split-seed", type=int, default=12)
    ap.add_argument("--n-test", type=int, default=104)
    ap.add_argument("--n-val", type=int, default=104)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-plot", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--u-norm", choices=["minmax", "none"], default="minmax",
                    help="minmax: divide by u_scale_max for the scaled cols (matches user config). "
                         "none: feed u_seq raw — try this if minmax gives bad R² and Bob's training "
                         "didn't pre-normalize.")
    ap.add_argument("--mm-channel", type=int, default=3,
                    help="Channel index of mm in y_seq full-state (default 3 = mm).")
    ap.add_argument("--p-channel", type=int, default=4)
    ap.add_argument("--pm-channel", type=int, default=5)
    ap.add_argument("--dna-control-idx", type=int, default=2,
                    help="Column in u_seq that holds DNA c (default 2).")
    ap.add_argument("--use-test", action="store_true",
                    help="Evaluate on test split (default). Use --use-val for val instead.")
    ap.add_argument("--use-val", action="store_true")
    ap.add_argument("--use-train", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"[info] device={args.device}  ckpt={args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    meta = ckpt.get("meta", {"in_u": 12, "hidden": 400})
    state_dict = ckpt.get("state_dict", ckpt)

    in_u = int(meta.get("in_u", 12))
    hidden = int(meta.get("hidden", 400))
    num_layers = 2
    print(f"[info] meta: in_u={in_u} hidden={hidden} layers={num_layers}")

    model = GRU_model_latent_decay_simple_fb_matO(
        in_u=in_u, hidden=hidden, num_layers=num_layers, dropout=0.2
    ).to(args.device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[info] load_state_dict: missing={list(missing)} unexpected={list(unexpected)}")

    print(f"[info] dataset={args.dataset}")
    d = np.load(args.dataset, allow_pickle=True)
    y0_all = d["y0"].astype(np.float32)        # (N, 7)
    u_seq_all = d["u_seq"].astype(np.float32)  # (N, K, 12)
    y_seq_all = d["y_seq"].astype(np.float32)  # (N, K, 7)
    t_obs = d["t_obs"].astype(np.float32)
    dt = np.diff(t_obs).astype(np.float32)     # (K,)
    lengths = d["lengths"].astype(np.int64) if "lengths" in d.files else np.full(y0_all.shape[0], y_seq_all.shape[1], dtype=np.int64)
    u_scale_max = d["u_scale_max"].astype(np.float32) if "u_scale_max" in d.files else None
    u_scaled_cols = d["u_scaled_cols"].astype(str) if "u_scaled_cols" in d.files else None
    control_names = d["control_names"].astype(str) if "control_names" in d.files else None

    # u normalization to match Bob's effective input scale.
    u_seq_norm = u_seq_all.copy()
    if args.u_norm == "minmax" and u_scale_max is not None and u_scaled_cols is not None and control_names is not None:
        ctrl_list = list(control_names)
        for c, vmax in zip(u_scaled_cols, u_scale_max):
            if c in ctrl_list and vmax > 0:
                j = ctrl_list.index(c)
                u_seq_norm[:, :, j] = u_seq_all[:, :, j] / float(vmax)
        print(f"[info] u_norm=minmax: divided {len(u_scaled_cols)} cols by u_scale_max")
    else:
        print(f"[info] u_norm={args.u_norm} (raw)")

    # Stratified split mirroring sp12 (stratify_targets=[3,5], 5 bins).
    train_idx, val_idx, test_idx = stratified_split(
        y_seq_all, lengths, n_test=args.n_test, n_val=args.n_val,
        seed=args.split_seed, n_bins=5, targets=(args.mm_channel, args.pm_channel),
    )
    if args.use_train:
        eval_idx, eval_name = train_idx, "train"
    elif args.use_val:
        eval_idx, eval_name = val_idx, "val"
    else:
        eval_idx, eval_name = test_idx, "test"
    print(f"[info] split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}  -> using {eval_name} (n={len(eval_idx)})")

    np.savez(out_dir / "split.npz",
             train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
             split_seed=np.array([args.split_seed]))

    obs_idx_3 = [args.mm_channel, args.p_channel, args.pm_channel]

    model.eval()
    with torch.no_grad():
        x0 = torch.from_numpy(y0_all[eval_idx][:, obs_idx_3]).to(args.device)
        u = torch.from_numpy(u_seq_norm[eval_idx]).to(args.device)
        dna_raw = u[:, :, args.dna_control_idx:args.dna_control_idx + 1]
        y_full = y_seq_all[eval_idx]                                # (B, K, 7)
        y3 = torch.from_numpy(y_full[:, :, obs_idx_3]).to(args.device)
        B, K, _ = u.shape
        dt_t = torch.from_numpy(dt).to(args.device).unsqueeze(0).expand(B, K).contiguous()

        pred, params = model(x0, u, dna_raw, dt_t, y3, teacher_forcing=False)
        pred = pred.cpu().numpy()  # (B, K, 3): [mm, p, pm]

    L = lengths[eval_idx]
    end = np.clip(L - 1, 0, K - 1)
    rows = np.arange(len(eval_idx))

    pm_true_final = y_full[rows, end, args.pm_channel]
    pm_pred_final = pred[rows, end, 2]

    # mm peak: max over valid timesteps.
    mm_true_seq = y_full[:, :, args.mm_channel]
    mm_pred_seq = pred[:, :, 0]
    mask = np.arange(K)[None, :] < L[:, None]
    mm_true_peak = np.where(mask, mm_true_seq, -np.inf).max(axis=1)
    mm_pred_peak = np.where(mask, mm_pred_seq, -np.inf).max(axis=1)

    r2_pm = r2_score(pm_true_final, pm_pred_final)
    r2_mm = r2_score(mm_true_peak, mm_pred_peak)
    print(f"[result] {eval_name} n={len(eval_idx)}  R² pm_final={r2_pm:.4f}  R² mm_peak={r2_mm:.4f}")

    run_tag = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.split_seed}_{eval_name}"
    csv_path = out_dir.parent / "r2_cache.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["run", "ckpt", "split_seed", "split", "n", "u_norm", "r2_pm_final", "r2_mm_peak"])
        w.writerow([run_tag, Path(args.ckpt).name, args.split_seed, eval_name,
                    len(eval_idx), args.u_norm, f"{r2_pm:.6f}", f"{r2_mm:.6f}"])
    print(f"[info] appended to {csv_path}")

    np.savez(out_dir / "preds.npz",
             eval_idx=eval_idx, lengths=L, t_obs=t_obs,
             pred_mm=pred[:, :, 0], pred_p=pred[:, :, 1], pred_pm=pred[:, :, 2],
             true_mm=mm_true_seq, true_p=y_full[:, :, args.p_channel],
             true_pm=y_full[:, :, args.pm_channel],
             pm_true_final=pm_true_final, pm_pred_final=pm_pred_final,
             mm_true_peak=mm_true_peak, mm_pred_peak=mm_pred_peak)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_plot = min(args.n_plot, len(eval_idx))
        sel = np.linspace(0, len(eval_idx) - 1, n_plot).astype(int)
        for j, k in enumerate(sel):
            L_k = int(L[k])
            t = t_obs[:L_k]
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
            ax[0].plot(t, mm_true_seq[k, :L_k], "k-", label="true mm")
            ax[0].plot(t, pred[k, :L_k, 0], "r--", label="pred mm")
            ax[0].set_title(f"mm  idx={int(eval_idx[k])}")
            ax[0].legend()
            ax[1].plot(t, y_full[k, :L_k, args.pm_channel], "k-", label="true pm")
            ax[1].plot(t, pred[k, :L_k, 2], "r--", label="pred pm")
            ax[1].set_title(f"pm  idx={int(eval_idx[k])}")
            ax[1].legend()
            fig.tight_layout()
            fig.savefig(plots_dir / f"pred_vs_true_{j:03d}_idx{int(eval_idx[k])}.png", dpi=110)
            plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
        ax[0].scatter(pm_true_final, pm_pred_final, s=14, alpha=0.7)
        lim = [min(pm_true_final.min(), pm_pred_final.min()),
               max(pm_true_final.max(), pm_pred_final.max())]
        ax[0].plot(lim, lim, "k--", lw=1)
        ax[0].set_xlabel("true pm_final")
        ax[0].set_ylabel("pred pm_final")
        ax[0].set_title(f"pm_final  R²={r2_pm:.3f}")
        ax[1].scatter(mm_true_peak, mm_pred_peak, s=14, alpha=0.7)
        lim = [min(mm_true_peak.min(), mm_pred_peak.min()),
               max(mm_true_peak.max(), mm_pred_peak.max())]
        ax[1].plot(lim, lim, "k--", lw=1)
        ax[1].set_xlabel("true mm_peak")
        ax[1].set_ylabel("pred mm_peak")
        ax[1].set_title(f"mm_peak  R²={r2_mm:.3f}")
        fig.tight_layout()
        fig.savefig(out_dir / "endpoint_r2.png", dpi=120)
        plt.close(fig)
        print(f"[info] saved {n_plot} trajectory plots and endpoint_r2.png to {out_dir}")
    except ImportError:
        print("[warn] matplotlib unavailable; skipping plots")


if __name__ == "__main__":
    main()
