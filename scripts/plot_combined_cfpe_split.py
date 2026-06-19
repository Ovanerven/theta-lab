"""Split variant of the combined cell-free figure: trajectories and parity as SEPARATE figures.

Same model and data as scripts/plot_combined_cfpe.py (best sLSTM M4 from the encoder zoo), but
instead of one composite figure it writes two, which read better when shown separately in the thesis:

  figures/cfpe_split/cfpe_trajectories.{pdf,png}
      2x2 grid: rows = (a) deoxygenated / (b) oxygenated, cols = mRNA / protein.
      The tube-opening event (O2 admitted) is marked on both channels of the oxygenated row.

  figures/cfpe_split/cfpe_parity.{pdf,png}
      Endpoint-protein parity for the best single model, (a) train and (b) test side by side
      (horizontal), wider than the stacked panels of the composite figure. The test panel reports
      R^2 separately for the deoxygenated and oxygenated protocols.

Usage:  python scripts/plot_combined_cfpe_split.py [run_dir]   (default: best sLSTM M4 from the encoder zoo)
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

OUT = ROOT / "figures" / "cfpe_split"; OUT.mkdir(parents=True, exist_ok=True)

# ======================== Figure 1: trajectories (2x2) ========================
MMd = 3
CH = [("mRNA", MMd, "#2e8b57"), ("Protein", PMd, "#4e79a7")]
figT, axesT = plt.subplots(2, 2, figsize=(3.7, 2.7))   # panel width matches the trajectory block of combined_cfpe
for ri, (sval, glab) in enumerate([(0, "deoxygenated"), (1, "oxygenated")]):
    samp = np.where(src_t == sval)[0]
    tmax = max(np.cumsum(dtT[i, :int(L[i])])[-1] for i in samp) / 60.0
    t_open = float(np.median([x for x in (uopen_time(i) for i in samp) if x is not None])) if sval == 1 else None
    for ci, (cname, ch, color) in enumerate(CH):
        ax = axesT[ri, ci]
        for i in samp:
            Li = int(L[i]); t = np.cumsum(dtT[i, :Li]) / 60.0
            ax.plot(t, yT[i, :Li, ch] - yT[i, :Li, ch].min(), color=color, lw=0.3, alpha=0.15)
        ax.set_xlim(0, tmax * 1.02); ax.set_ylim(0, None); ax.set_xlabel("time (min)")
        if ci == 0:
            ax.set_ylabel("conc.\\ (nM)")
            ax.set_title(glab, x=-0.30, ha="left", fontsize=9.5, fontweight="bold", pad=4)
        ax.text(0.05, 0.92, cname, transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
                fontweight="bold", bbox=dict(facecolor=color, alpha=0.18, edgecolor="none", boxstyle="round,pad=0.25"))
        if t_open is not None:
            xo = t_open + UOPEN_SHIFT_MIN
            ax.axvline(xo, color=UOPEN, ls="--", lw=1.0, alpha=0.9)
            ax.text(xo, ax.get_ylim()[1] * 0.97, " tube opened", color=UOPEN, fontsize=6, va="top", ha="left")
figT.tight_layout(h_pad=0.6, w_pad=0.6)
figT.savefig(str(OUT / "cfpe_trajectories.pdf"), bbox_inches="tight")
figT.savefig(str(OUT / "cfpe_trajectories.png"), bbox_inches="tight", dpi=300)
plt.close(figT)

# ======================== Figure 2: parity, train | test (compact, parity_protein.py style) ========================
# Dimensions follow figures/parity/parity_protein.{pdf,png}: slightly wider, slightly shorter, a touch larger font.
# Per-protocol R^2 sits in the top-left legend box; the Train/Test tag sits in a gray box bottom-left.
with plt.rc_context({"font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5, "legend.fontsize": 7.5}):
    figP, axesP = plt.subplots(1, 2, figsize=(5.0, 1.95))
    handles = None
    for ai, (name, idx) in enumerate([("Train", split["train_idx"]), ("Test", split["test_idx"])]):
        ax = axesP[ai]
        m, p, src = endpoints(idx)
        hi = max(np.r_[m, p].max(), 1.0); lim = (-0.03*hi, 1.05*hi)
        ax.plot(lim, lim, "--", color="0.25", lw=1.0, zorder=1)
        hh = []
        for s, c, nm in [(0, SEALED, "deoxygenated"), (1, OPENED, "oxygenated")]:
            k = src == s
            hh.append(ax.scatter(m[k], p[k], s=11, c=c, alpha=0.5, linewidths=0, zorder=2, label=nm))
        handles = hh   # identical convention in both panels -> one shared legend
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
        ax.set_xlabel("Measured protein (nM)"); ax.set_ylabel("Predicted protein (nM)")
        # short-form R^2 in the empty top-left corner (blocks no points): overall, then per protocol
        ax.text(0.05, 0.94, rf"$R^2_{{\mathrm{{all}}}}={r2(m,p):.2f}$", transform=ax.transAxes,
                va="top", ha="left", fontsize=8, color="black",
                bbox=dict(facecolor="0.85", edgecolor="none", boxstyle="round,pad=0.25"))
        ax.text(0.05, 0.74, rf"$R^2_{{\mathrm{{deox}}}}={r2(m,p,src==0):.2f}$", transform=ax.transAxes,
                va="top", ha="left", fontsize=8, color="black")
        ax.text(0.05, 0.58, rf"$R^2_{{\mathrm{{ox}}}}={r2(m,p,src==1):.2f}$", transform=ax.transAxes,
                va="top", ha="left", fontsize=8, color="black")
        # protein-colored box (matches the "Protein" channel box in the trajectory figure): signals this is protein
        ax.text(0.95, 0.06, name, transform=ax.transAxes, va="bottom", ha="right", fontsize=8.5,
                fontweight="bold", bbox=dict(facecolor=PROT, alpha=0.18, edgecolor="none", boxstyle="round,pad=0.25"))
    # shared legend above both panels, outside the axes
    figP.legend(handles, ["deoxygenated", "oxygenated"], loc="lower center",
                bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                handletextpad=0.3, columnspacing=1.4)
    figP.tight_layout(w_pad=0.3)
    figP.savefig(str(OUT / "cfpe_parity.pdf"), bbox_inches="tight")
    figP.savefig(str(OUT / "cfpe_parity.png"), bbox_inches="tight", dpi=300)
    plt.close(figP)

print(f"wrote {OUT}/cfpe_trajectories.{{pdf,png}} and {OUT}/cfpe_parity.{{pdf,png}}")
