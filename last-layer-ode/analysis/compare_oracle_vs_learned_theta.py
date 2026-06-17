"""Compare oracle per-step theta vs a trained model's learned theta(t).

Loads a trained run via rebuild_model_from_experiment, runs the model over ALL
samples in dataset order (mirroring plot_predictions' gate+lift pipeline), pulls
the per-step theta(t) in physical units, and aligns it sample-for-sample with an
oracle theta array (same dataset, same order). Writes a comparison npz + a
per-parameter figure.

Run from the last-layer-ode/ directory:
    python analysis/compare_oracle_vs_learned_theta.py \
        --exp ../experiments_final/FINAL/FINAL_coarse_encoder_zoo_light/20260607_180650_EZ_slstm_s2 \
        --oracle ../results/oracle_theta_fit/results.npz \
        --out ../results/oracle_vs_learned_M4_slstm_s2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # last-layer-ode/

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_diagnostics import (
    rebuild_model_from_experiment, _maybe_lift, _filter_model_kwargs, load_yaml,
)
from train import collate, collate_varlen
from scaffolds import SCAFFOLDS


def extract_learned_theta(exp_dir: Path, device: torch.device):
    model, ds, state_names, param_names, lift_info = rebuild_model_from_experiment(exp_dir, device)
    model.eval()
    raw_ds = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    collate_fn = collate_varlen if getattr(raw_ds, "variable_length", False) else collate
    cfg = load_yaml(exp_dir / "config.yaml")
    u_transform = str(cfg.get("u_transform", "none"))
    y_transform = str(cfg.get("y_transform", "none"))

    loader = torch.utils.data.DataLoader(
        ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_fn)

    thetas, lengths = [], []
    from metrics.endpoint_r2 import _apply_channel_min_gate as _gate
    for batch in loader:
        y0, u_seq, y_seq, batch_lengths = batch[0], batch[1], batch[2], batch[3]
        dt_seq = batch[5]
        y0, u_seq, y_seq, dt_seq = y0.to(device), u_seq.to(device), y_seq.to(device), dt_seq.to(device)
        if batch_lengths is not None:
            batch_lengths = batch_lengths.to(device)
        if bool(cfg.get("subtract_channel_min", False)):
            cols = cfg.get("subtract_channel_min_cols", None)
            cols = [int(c) for c in cols] if cols is not None else None
            y0, y_seq = _gate(y0, y_seq, cols, batch_lengths)
        y0, y_seq = _maybe_lift(y0, y_seq, lift_info or {})

        if lift_info:
            obs_idx = torch.tensor(lift_info["scaffold_obs_idx"], device=y0.device, dtype=torch.long)
        else:
            obs_idx = torch.arange(y0.shape[-1], device=y0.device)
        mk = {"y_seq": None, "teacher_forcing": False,
              "u_transform": u_transform, "y_transform": y_transform}
        if model.__class__.__name__ == "OdeTransformerGrouped" and batch_lengths is not None:
            mk["lengths"] = batch_lengths
        with torch.no_grad():
            _pred, theta, _beta = model(y0, u_seq, dt_seq, obs_idx, **_filter_model_kwargs(model, mk))
        thetas.append(theta.cpu().numpy())
        lengths.append(batch_lengths.cpu().numpy() if batch_lengths is not None
                       else np.full(theta.shape[0], theta.shape[1]))
    Tmax = max(t.shape[1] for t in thetas)
    D = thetas[0].shape[2]
    padded = []
    for t in thetas:
        if t.shape[1] < Tmax:
            pad = np.full((t.shape[0], Tmax - t.shape[1], D), np.nan, dtype=t.dtype)
            t = np.concatenate([t, pad], axis=1)
        padded.append(t)
    learned = np.concatenate(padded, axis=0)       # (N, Tmax, D)
    lengths = np.concatenate(lengths, axis=0)
    scaffold = SCAFFOLDS[cfg["scaffold"]]
    return learned, lengths, param_names, np.array(scaffold.theta_lo_vec), np.array(scaffold.theta_hi_vec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cpu")

    learned, m_len, param_names, lo, hi = extract_learned_theta(Path(args.exp), device)
    orc = np.load(args.oracle, allow_pickle=True)
    oth = orc["thetas"].astype(np.float64)        # (N, To, D)
    o_len = orc["lengths"].astype(int)
    src = orc["source_label"]

    print(f"learned theta {learned.shape}  oracle theta {oth.shape}")
    N = min(learned.shape[0], oth.shape[0])
    T = min(learned.shape[1], oth.shape[1])
    D = learned.shape[2]
    learned = learned[:N, :T, :].astype(np.float64)
    oth = oth[:N, :T, :]
    L = np.minimum(m_len[:N], o_len[:N])
    src = src[:N]

    # valid-step mask (exclude padding)
    step = np.arange(T)[None, :]
    mask = step < L[:, None]                       # (N, T)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, D, figsize=(4 * D, 8))
    rows = []
    for j in range(D):
        lv = learned[:, :, j][mask]
        ov = oth[:, :, j][mask]
        eps = 1e-30
        llog, olog = np.log10(lv + eps), np.log10(ov + eps)
        r = np.corrcoef(llog, olog)[0, 1]
        med_ratio = np.median(lv) / (np.median(ov) + eps)
        rows.append((param_names[j], np.median(ov), np.median(lv), med_ratio, r))

        # top: 2D log-log density of learned vs oracle
        ax = axes[0][j]
        ax.hexbin(olog, llog, gridsize=40, cmap="viridis", bins="log")
        lim = [min(olog.min(), llog.min()), max(olog.max(), llog.max())]
        ax.plot(lim, lim, "r--", lw=1)
        for b in (lo[j], hi[j]):
            ax.axvline(np.log10(b), color="orange", ls=":", lw=0.8)
            ax.axhline(np.log10(b), color="cyan", ls=":", lw=0.8)
        ax.set_xlabel(f"oracle log10 {param_names[j]}")
        ax.set_ylabel(f"learned log10 {param_names[j]}")
        ax.set_title(f"{param_names[j]}  r={r:.2f}")

        # bottom: time-median curves
        ax2 = axes[1][j]
        om = np.array([np.median(oth[mask[:, t], t, j]) for t in range(T)])
        lm = np.array([np.median(learned[mask[:, t], t, j]) for t in range(T)])
        ax2.plot(om, label="oracle", color="C0")
        ax2.plot(lm, label="learned", color="C3")
        ax2.axhline(lo[j], color="orange", ls=":", lw=0.8)
        ax2.axhline(hi[j], color="cyan", ls=":", lw=0.8)
        ax2.set_yscale("log")
        ax2.set_title(f"{param_names[j]} time-median")
        ax2.set_xlabel("step")
        if j == 0:
            ax2.legend()
    fig.tight_layout()
    fig.savefig(out / "theta_compare.png", dpi=130)

    print(f"\n{'param':10} {'oracle_med':>12} {'learned_med':>12} {'L/O ratio':>10} {'logcorr':>8}")
    for nm, om, lm, rt, r in rows:
        print(f"{nm:10} {om:12.4g} {lm:12.4g} {rt:10.3g} {r:8.3f}")
    np.savez_compressed(out / "compare.npz", learned=learned, oracle=oth, mask=mask,
                        lo=lo, hi=hi, param_names=np.array(param_names), src=src)
    print(f"\nWrote {out}/theta_compare.png and compare.npz")


if __name__ == "__main__":
    main()
