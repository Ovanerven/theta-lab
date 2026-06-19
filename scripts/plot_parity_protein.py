"""Parity plot: predicted vs measured endpoint protein (nM), Train | Test panels, R² annotated.

For the paper. Defaults to the best M4 dense model (NG_plain, argmax test protein R²); pass a run
dir to override. Open-loop endpoint protein (gated, like the metrics), one point per trajectory.
No caption (add in LaTeX). Output: figures/parity/parity_protein.{pdf,png}

Usage: python scripts/plot_parity_protein.py [run_dir]
"""
import sys, glob, csv
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/olivervanerven/Documents/Thesis/theta-lab"); LLO = ROOT / "last-layer-ode"
sys.path.insert(0, str(LLO))
from plot_diagnostics import (rebuild_model_from_experiment, _maybe_lift, _filter_model_kwargs, load_yaml)
from train import collate, collate_varlen
from metrics.endpoint_r2 import _apply_channel_min_gate as _gate

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
})

# pick run: arg, else best M4 NG_plain by test protein R²
if len(sys.argv) > 1:
    EXP = Path(sys.argv[1])
else:
    cand = []
    for d in glob.glob(str(ROOT / "experiments_final/FINAL/scaffold_ladder/*NG_plain*")):
        rc = Path(d) / "r2_cache.csv"
        if rc.exists():
            r = list(csv.DictReader(open(rc)))[-1]
            cand.append((float(r["r2_protein_final"]), d))
    EXP = Path(max(cand)[1])
print("model:", EXP.name)

dev = torch.device("cpu")
model, ds, *_r = rebuild_model_from_experiment(EXP, dev); model.eval()
lift_info = _r[2] or {}; cfg = load_yaml(EXP / "config.yaml")
raw = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
PM = int(lift_info["scaffold_obs_idx"][1]) if lift_info else 2     # protein obs index
split = np.load(EXP / "split.npz")

def endpoints(idxs):
    sub = torch.utils.data.Subset(raw, list(idxs))
    cf = collate_varlen if getattr(raw, "variable_length", False) else collate
    b = next(iter(torch.utils.data.DataLoader(sub, batch_size=len(idxs), shuffle=False, collate_fn=cf)))
    y0, u, y, Ls = b[0], b[1], b[2], b[3]
    dt = b[5] if len(b) >= 6 else torch.from_numpy(raw.dt[:u.shape[1]])[None].expand(y0.shape[0], -1)
    if bool(cfg.get("subtract_channel_min", False)):
        cols = cfg.get("subtract_channel_min_cols"); cols = [int(c) for c in cols] if cols else None
        y0, y = _gate(y0, y, cols, Ls)
    y0, y = _maybe_lift(y0, y, lift_info)
    oi = torch.tensor(lift_info["scaffold_obs_idx"], dtype=torch.long) if lift_info else torch.arange(y0.shape[-1])
    mk = {"y_seq": None, "teacher_forcing": False, "u_transform": str(cfg.get("u_transform", "none")),
          "y_transform": str(cfg.get("y_transform", "none"))}
    with torch.no_grad():
        pred, _, _ = model(y0, u, dt, oi, **_filter_model_kwargs(model, mk))
    pred = pred.cpu().numpy(); Ls = Ls.cpu().numpy()
    meas = np.array([y[j, Ls[j]-1, PM].item() for j in range(len(idxs))])
    prd  = np.array([pred[j, Ls[j]-1, PM] for j in range(len(idxs))])
    src  = np.array([int(raw.source_idx[i]) for i in idxs])      # 0=old/sealed, 1=new/opened
    return meas, prd, src

def r2(t, p, mask=None):
    if mask is not None: t, p = t[mask], p[mask]
    ss = np.sum((t - t.mean())**2); return 1 - np.sum((t - p)**2)/ss

SEALED, OPENED = "#4e79a7", "#f28e2b"          # sealed (old) blue, opened (new) orange
panels = [("Train", split["train_idx"]), ("Test", split["test_idx"])]
data = {name: endpoints(idx) for name, idx in panels}
hi = max(np.concatenate([np.r_[m, p] for m, p, _ in data.values()]).max(), 1.0)
lim = (-0.03*hi, 1.05*hi)

def scat(ax, m, p, src):
    for s, c, lab in [(0, SEALED, "deoxygenated"), (1, OPENED, "oxygenated")]:
        k = src == s
        ax.scatter(m[k], p[k], s=16, c=c, alpha=0.55, linewidths=0, zorder=2, label=lab)

fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.15))
for ax, (name, _) in zip(axes, panels):
    m, p, src = data[name]
    ax.plot(lim, lim, "--", color="0.25", lw=1.0, zorder=1)
    scat(ax, m, p, src)
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_xlabel("Measured protein (nM)"); ax.set_ylabel("Predicted protein (nM)")
    ax.text(0.04, 0.97, name, transform=ax.transAxes, va="top", ha="left", fontstyle="italic", fontsize=9)
    if name == "Train":
        ax.text(0.04, 0.87, f"$R^2 = {r2(m, p):.3f}$", transform=ax.transAxes, va="top", fontsize=8.5)
    else:   # Test: split R² by experiment type, color-matched, + the sealed/opened legend here
        ax.text(0.04, 0.87, f"deoxygenated $R^2 = {r2(m, p, src==0):.3f}$", transform=ax.transAxes,
                va="top", fontsize=8.5, color=SEALED)
        ax.text(0.04, 0.76, f"oxygenated $R^2 = {r2(m, p, src==1):.3f}$", transform=ax.transAxes,
                va="top", fontsize=8.5, color=OPENED)
        ax.legend(loc="lower right", frameon=False, fontsize=7.5, handletextpad=0.2, borderpad=0.2)
fig.tight_layout()
out = ROOT / "figures" / "parity"; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "parity_protein.pdf", bbox_inches="tight")
fig.savefig(out / "parity_protein.png", bbox_inches="tight", dpi=300)
mt, pt, st = data["Test"]
print(f"wrote {out}/parity_protein.{{pdf,png}}  | Train R²={r2(*data['Train'][:2]):.3f} | "
      f"Test all={r2(mt,pt):.3f} sealed={r2(mt,pt,st==0):.3f} opened={r2(mt,pt,st==1):.3f}")
