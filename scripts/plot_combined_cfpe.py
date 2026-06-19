"""Combined cell-free figure (first draft, to iterate on):
  LEFT column  : protein trajectories, (a) sealed (top) / (b) opened (bottom, u_open marker).
  RIGHT column : endpoint-protein parity for the best single model, (c) train (top) / (d) test (bottom).
The model is trained on both protocols; the test panel reports R^2 separately for sealed/opened.

Output: figures/combined_cfpe.{pdf,png}
Usage:  python scripts/plot_combined_cfpe.py [run_dir]   (default: best sLSTM M4 from the encoder zoo)
"""
import sys, glob, csv
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path("/Users/olivervanerven/Documents/Thesis/theta-lab"); LLO = ROOT / "last-layer-ode"
sys.path.insert(0, str(LLO))
from plot_diagnostics import (rebuild_model_from_experiment, _maybe_lift, _filter_model_kwargs, load_yaml)
from train import collate, collate_varlen
from metrics.endpoint_r2 import _apply_channel_min_gate as _gate

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
})
SEALED, OPENED, PROT, UOPEN = "#4e79a7", "#f28e2b", "#4e79a7", "#c44e52"
UOPEN_SHIFT_MIN = 0.0   # marker at the true event time (22 h, = step 150); coincides with the mRNA jump

# ---------- model: best sLSTM M4 by test protein R^2 (matches the parity figure) ----------
if len(sys.argv) > 1:
    EXP = Path(sys.argv[1])
else:
    cand = []
    for d in glob.glob(str(ROOT / "experiments_final/FINAL/FINAL_coarse_encoder_zoo_light/*EZ_slstm*")):
        rc = Path(d) / "r2_cache.csv"
        if rc.exists():
            cand.append((float(list(csv.DictReader(open(rc)))[-1]["r2_protein_final"]), d))
    EXP = Path(max(cand)[1])
print("model:", EXP.name)
dev = torch.device("cpu")
model, ds, *_r = rebuild_model_from_experiment(EXP, dev); model.eval()
lift_info = _r[2] or {}; cfg = load_yaml(EXP / "config.yaml")
raw = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
PMs = int(lift_info["scaffold_obs_idx"][1]) if lift_info else 2     # protein, scaffold space
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
    meas = np.array([y[j, Ls[j]-1, PMs].item() for j in range(len(idxs))])
    prd  = np.array([pred[j, Ls[j]-1, PMs] for j in range(len(idxs))])
    src  = np.array([int(raw.source_idx[i]) for i in idxs])
    return meas, prd, src
def r2(t, p, m=None):
    if m is not None: t, p = t[m], p[m]
    ss = np.sum((t - t.mean())**2); return 1 - np.sum((t - p)**2)/ss

# ---------- trajectory data (dataset space; protein = col 5) ----------
z = np.load(ROOT / "datasets/cell-free/txtl_native_real_only_coarsenold.npz", allow_pickle=True)
yT, L, dtT = z["y_seq"], z["lengths"], z["dt_per_sample"]
src_t = np.array([1 if str(s) == "new" else 0 for s in z["source_label"]]); PMd = 5
uu = z["u_seq"]; OI = [str(c) for c in z["control_names"]].index("u_open")
def uopen_time(i):
    Li = int(L[i]); nz = np.nonzero(uu[i, :Li, OI] > 0)[0]
    return float(np.cumsum(dtT[i, :nz[0]+1])[-1] / 60.0) if len(nz) else None

# ---------- layout: (a) sealed [mRNA,protein] | (b) opened [mRNA,protein] | (c) parity [train,test] ----------
MMd = 3
CH = [("mRNA", MMd, "#2e8b57"), ("Protein", PMd, "#4e79a7")]
fig = plt.figure(figsize=(6.7, 3.2))
outer = GridSpec(1, 2, figure=fig, width_ratios=[1.12, 0.66], wspace=0.02)
gsL = outer[0].subgridspec(2, 2, wspace=0.20, hspace=0.40)    # trajectories: rows protocol, cols channel
gsR = outer[1].subgridspec(2, 1, hspace=0.40)                 # parity: train / test

# trajectories (left two columns)
for ri, (sval, glab) in enumerate([(0, "(a) deoxygenated"), (1, "(b) oxygenated")]):
    samp = np.where(src_t == sval)[0]
    tmax = max(np.cumsum(dtT[i, :int(L[i])])[-1] for i in samp) / 60.0
    t_open = float(np.median([x for x in (uopen_time(i) for i in samp) if x is not None])) if sval == 1 else None
    for ci, (cname, ch, color) in enumerate(CH):
        ax = fig.add_subplot(gsL[ri, ci])
        for i in samp:
            Li = int(L[i]); t = np.cumsum(dtT[i, :Li]) / 60.0
            ax.plot(t, yT[i, :Li, ch] - yT[i, :Li, ch].min(), color=color, lw=0.3, alpha=0.15)
        ax.set_xlim(0, tmax * 1.02); ax.set_ylim(0, None); ax.set_xlabel("time (min)")
        if ci == 0:
            ax.set_ylabel("conc.\\ (nM)")
            ax.set_title(glab, x=-0.32, ha="left", fontsize=9.5, fontweight="bold", pad=4)  # aligns with (c)
        # green/blue inset box naming the channel, as in the standalone trajectory figure
        ax.text(0.05, 0.92, cname, transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
                fontweight="bold", bbox=dict(facecolor=color, alpha=0.18, edgecolor="none", boxstyle="round,pad=0.25"))
        if t_open is not None:
            xo = t_open + UOPEN_SHIFT_MIN
            ax.axvline(xo, color=UOPEN, ls="--", lw=1.0, alpha=0.9)
            ax.text(xo, ax.get_ylim()[1] * 0.97, " tube opened", color=UOPEN, fontsize=6, va="top", ha="left")

# parity (right block)
for ri, (name, idx) in enumerate([("train", split["train_idx"]), ("test", split["test_idx"])]):
    ax = fig.add_subplot(gsR[ri, 0])
    m, p, src = endpoints(idx)
    hi = max(np.r_[m, p].max(), 1.0); lim = (-0.03*hi, 1.05*hi)
    ax.plot(lim, lim, "--", color="0.25", lw=0.9, zorder=1)
    for s, c, nm in [(0, SEALED, "deoxygenated"), (1, OPENED, "oxygenated")]:
        k = src == s
        ax.scatter(m[k], p[k], s=9, c=c, alpha=0.5, linewidths=0, zorder=2,
                   label=f"{nm}: $R^2{{=}}{r2(m,p,src==s):.2f}$")
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_xlabel("measured (nM)"); ax.set_ylabel("predicted (nM)")
    if ri == 0: ax.set_title("(c) predictions", loc="left", fontsize=9.5, fontweight="bold", pad=4)
    # R^2 legend in the empty top-left corner; train/test in a gray box bottom-right (mRNA/Protein style)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="0.85", facecolor="white",
              fontsize=5.8, handletextpad=0.25, borderpad=0.3, labelspacing=0.25)
    ax.text(0.95, 0.06, name, transform=ax.transAxes, va="bottom", ha="right", fontsize=7.5,
            fontweight="bold", bbox=dict(facecolor="0.5", alpha=0.22, edgecolor="none", boxstyle="round,pad=0.25"))

out = ROOT / "figures" / "combined_cfpe"
fig.savefig(str(out) + ".pdf", bbox_inches="tight")
fig.savefig(str(out) + ".png", bbox_inches="tight", dpi=300)
print(f"wrote figures/combined_cfpe.{{pdf,png}}")
