"""Coarse cell-free dataset — mRNA + protein trajectories, split by experiment type.

2x2: row (a) = sealed (tube closed, ~22 h); row (b) = opened (tube reopened at steady state, ~45 h).
Columns = mRNA | protein. Trajectories are baseline-gated (per-sample channel-min subtraction, as in
train/eval). Shared axes so the differing record lengths are visible. No caption (add in LaTeX).
Output: figures/coarse_data/coarse_data_trajectories.{pdf,png}
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
})

ROOT = Path("/Users/olivervanerven/Documents/Thesis/theta-lab")
z = np.load(ROOT / "datasets/cell-free/txtl_native_real_only_coarsenold.npz", allow_pickle=True)
y, L, dt = z["y_seq"], z["lengths"], z["dt_per_sample"]
src = np.array([1 if str(s) == "new" else 0 for s in z["source_label"]])   # 0=old/sealed, 1=new/opened
MM, PM = 3, 5

# u_open: the tube-opening event (O2 admitted) -> only fires in opened experiments. Mark its time.
u = z["u_seq"]; OI = [str(c) for c in z["control_names"]].index("u_open")
def uopen_time(i):
    Li = int(L[i]); nz = np.nonzero(u[i, :Li, OI] > 0)[0]
    return float(np.cumsum(dt[i, :nz[0] + 1])[-1] / 60.0) if len(nz) else None
UOPEN = "#c44e52"   # tube-opening marker colour
UOPEN_SHIFT_MIN = 60.0   # cosmetic nudge of the marker (min); 0 = true event time (22 h)

ROWS = [(0, "(a) deoxygenated"), (1, "(b) oxygenated")]
COLS = [("mRNA", MM, "#2e8b57"), ("Protein", PM, "#4e79a7")]
# independent axes: each protocol gets its OWN time range (sealed shorter, opened longer) and each
# panel autoscales y, so neither protocol is squished by the other's scale.
fig, axes = plt.subplots(2, 2, figsize=(5.4, 3.5))
for ri, (sval, label) in enumerate(ROWS):
    samp = np.where(src == sval)[0]
    tmax = max(np.cumsum(dt[i, :int(L[i])])[-1] for i in samp) / 60.0    # this protocol's longest run
    # opened experiments: time of the tube-opening event (consistent across samples)
    t_open = None
    if sval == 1:
        ts = [uopen_time(i) for i in samp]; ts = [x for x in ts if x is not None]
        t_open = float(np.median(ts)) if ts else None
    for ci, (name, ch, color) in enumerate(COLS):
        ax = axes[ri, ci]
        for i in samp:
            Li = int(L[i]); t = np.cumsum(dt[i, :Li]) / 60.0
            seg = y[i, :Li, ch] - y[i, :Li, ch].min()            # channel-min gate (as in train/eval)
            ax.plot(t, seg, color=color, lw=0.3, alpha=0.18)
        ax.set_xlim(0, tmax * 1.02); ax.set_ylim(0, None)        # fill the row's own time axis
        if t_open is not None:                                   # mark the tube opening
            xo = t_open + UOPEN_SHIFT_MIN
            ax.axvline(xo, color=UOPEN, ls="--", lw=1.2, alpha=0.9, zorder=5)
            ax.text(xo, ax.get_ylim()[1] * 0.97, " tube opened", color=UOPEN,
                    fontsize=7, va="top", ha="left", fontweight="bold")
        if ci == 0: ax.set_ylabel("Concentration (nM)")
        ax.set_xlabel("Time (min)")
        ax.text(0.035, 0.94, name, transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.18, edgecolor="none", boxstyle="round,pad=0.2"))
    axes[ri, 0].text(-0.30, 1.04, label, transform=axes[ri, 0].transAxes,
                     va="bottom", ha="left", fontsize=9, fontweight="bold")
fig.tight_layout(h_pad=1.4)
out = ROOT / "figures" / "coarse_data"; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "coarse_data_trajectories.pdf", bbox_inches="tight")
fig.savefig(out / "coarse_data_trajectories.png", bbox_inches="tight", dpi=300)
print(f"wrote {out}/coarse_data_trajectories.{{pdf,png}}  | sealed n={int((src==0).sum())} opened n={int((src==1).sum())}")
