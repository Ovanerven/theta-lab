"""Per-trajectory protein comparison: E21 (endpoint+SGDR, protein_old 0.747) vs the
baseline sLSTM seed-2 (protein_old 0.721). Finds OLD samples where E21 predicts the
protein (pm) endpoint markedly better, and dumps trajectory + theta panels to show why.

Run from last-layer-ode/:
  python analysis/compare_e21_vs_baseline_traj.py \
    --e21  ../experiments_goal/goal_wave/<...>_E21_redo \
    --base ../experiments_final/FINAL/FINAL_coarse_encoder_zoo_light/20260607_180650_EZ_slstm_s2 \
    --out  ../results/e21_vs_baseline_traj
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_diagnostics import rebuild_model_from_experiment, _maybe_lift, _filter_model_kwargs, load_yaml
from train import collate, collate_varlen
from metrics.endpoint_r2 import _apply_channel_min_gate as _gate


def run_model(exp_dir, device):
    """Return per-sample: pred pm traj, true pm traj, theta, lengths, source, dt — dataset order."""
    model, ds, state_names, param_names, lift_info = rebuild_model_from_experiment(exp_dir, device)
    model.eval()
    raw = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    cfn = collate_varlen if getattr(raw, "variable_length", False) else collate
    cfg = load_yaml(exp_dir / "config.yaml")
    ut, yt = str(cfg.get("u_transform", "none")), str(cfg.get("y_transform", "none"))
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=cfn)
    pm_idx = lift_info["scaffold_obs_idx"][1] if lift_info else 5   # pm = 2nd observed (P_fluor=2)

    preds, trues, thetas, lens = [], [], [], []
    for b in loader:
        y0, u, y, blen, dt = b[0], b[1], b[2], b[3], b[5]
        y0, u, y, dt = y0.to(device), u.to(device), y.to(device), dt.to(device)
        blen = blen.to(device) if blen is not None else None
        if bool(cfg.get("subtract_channel_min", False)):
            cols = cfg.get("subtract_channel_min_cols", None)
            cols = [int(c) for c in cols] if cols is not None else None
            y0, y = _gate(y0, y, cols, blen)
        y0, y = _maybe_lift(y0, y, lift_info or {})
        obs_idx = (torch.tensor(lift_info["scaffold_obs_idx"], device=y0.device, dtype=torch.long)
                   if lift_info else torch.arange(y0.shape[-1], device=y0.device))
        mk = {"y_seq": None, "teacher_forcing": False, "u_transform": ut, "y_transform": yt}
        with torch.no_grad():
            pred, theta, _ = model(y0, u, dt, obs_idx, **_filter_model_kwargs(model, mk))
        preds.append(pred[:, :, pm_idx].cpu().numpy())
        trues.append(y[:, :, pm_idx].cpu().numpy())
        thetas.append(theta.cpu().numpy())
        lens.append(blen.cpu().numpy() if blen is not None else np.full(pred.shape[0], pred.shape[1]))
    Tmax = max(p.shape[1] for p in preds)
    def pad(a):
        return np.concatenate([np.pad(x, ((0,0),(0,Tmax-x.shape[1])), constant_values=np.nan) for x in a], 0)
    Dn = thetas[0].shape[2]
    thpad = np.concatenate([np.pad(t, ((0,0),(0,Tmax-t.shape[1]),(0,0)), constant_values=np.nan) for t in thetas], 0)
    # source_idx: 0=old, 1=new, 2=synth, -1=unknown (ODEDataset, aligned to order)
    sidx = np.asarray(raw.source_idx) if getattr(raw, "source_idx", None) is not None else np.full(np.concatenate(lens).shape[0], -1)
    src = np.where(sidx == 0, "old", np.where(sidx == 1, "new", "other"))
    return pad(preds), pad(trues), thpad, np.concatenate(lens), src, param_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e21", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()
    dev = torch.device("cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    e_pred, e_true, e_th, lens, src, pnames = run_model(Path(args.e21), dev)
    b_pred, b_true, b_th, _, _, _ = run_model(Path(args.base), dev)
    N = min(len(lens), b_pred.shape[0]);

    # per-sample protein endpoint (value at L-1) error for each model
    rows = []
    for i in range(N):
        L = int(lens[i])
        t_end = e_true[i, L-1]
        e_end, b_end = e_pred[i, L-1], b_pred[i, L-1]
        rows.append((i, str(src[i]), t_end, e_end, b_end, abs(e_end - t_end), abs(b_end - t_end)))
    arr = np.array([(r[5], r[6]) for r in rows])  # (e_err, b_err)
    e_err, b_err = arr[:,0], arr[:,1]
    is_old = np.array([r[1] == "old" for r in rows])

    # OLD samples where E21 beats baseline most (b_err - e_err large positive)
    improve = b_err - e_err
    cand = np.where(is_old)[0]
    order = cand[np.argsort(-improve[cand])]
    top = order[:args.topk]

    print(f"OLD samples: {is_old.sum()} | E21 better on endpoint: {(improve[cand]>0).sum()}/{len(cand)}")
    print(f"median |err| old — E21 {np.median(e_err[cand]):.3g}  baseline {np.median(b_err[cand]):.3g}")
    print(f"\nTop {args.topk} OLD samples where E21 beats baseline (endpoint):")
    print(f"{'idx':>5} {'true_end':>10} {'E21_end':>10} {'base_end':>10} {'E21|err|':>9} {'base|err|':>9}")
    for i in top:
        r = rows[i]
        print(f"{r[0]:5d} {r[2]:10.3g} {r[3]:10.3g} {r[4]:10.3g} {r[5]:9.3g} {r[6]:9.3g}")

    # trajectory + theta panels for the top samples
    D = e_th.shape[2]
    fig, axes = plt.subplots(2, len(top), figsize=(3.2*len(top), 7), squeeze=False)
    for c, i in enumerate(top):
        L = int(lens[i])
        ax = axes[0][c]
        ax.plot(e_true[i, :L], 'k-', lw=2, label='true')
        ax.plot(b_pred[i, :L], 'C0--', lw=1.5, label='baseline')
        ax.plot(e_pred[i, :L], 'C3-', lw=1.5, label='E21')
        ax.set_title(f"idx {i} (old) pm"); ax.set_xlabel("step")
        if c == 0: ax.legend(fontsize=7)
        ax2 = axes[1][c]
        for j in range(D):
            ax2.plot(e_th[i, :L, j], lw=1.2, label=pnames[j] if c==0 else None)
        ax2.set_yscale("log"); ax2.set_title("E21 θ(t)"); ax2.set_xlabel("step")
        if c == 0: ax2.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out/"e21_wins_trajectories.png", dpi=130)
    np.savez_compressed(out/"compare.npz", e_pred=e_pred, b_pred=b_pred, true=e_true,
                        e_theta=e_th, lens=lens, src=src, improve=improve, top=top)
    print(f"\nWrote {out}/e21_wins_trajectories.png and compare.npz")


if __name__ == "__main__":
    main()
