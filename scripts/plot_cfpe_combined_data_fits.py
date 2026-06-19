"""Combined cell-free figure for the thesis: (a) measured trajectory data + (b) representative fits.

One matplotlib figure (not LaTeX subfigures) so the two 2x2 blocks share a gridspec and their
axis bottoms line up exactly, and the legend lives inside the figure.

  (a) measured trajectories  : rows = deoxygenated / oxygenated, cols = mRNA / protein
                               (tube-opening marked on the oxygenated row)
  (b) representative fits     : 3 good + 1 failure example (FAIL set), measured (solid) vs predicted (dashed)

Reuses the prediction cache from scripts/plot_cfps_examples.py (figures/cfps_examples/cache.npz), so no
model rebuild is needed.

Output: figures/cfpe_combined/cfpe_data_and_fits.{pdf,png}
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path("/Users/olivervanerven/Documents/Thesis/theta-lab")
OUT = ROOT / "figures" / "cfpe_combined"; OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6.5, "figure.dpi": 300, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "lines.linewidth": 1.0,
})
C_MRNA, C_PROT, UOPEN = "#2e8b57", "#4e79a7", "#c44e52"

# ---------------- left block: measured trajectory data ----------------
z = np.load(ROOT / "datasets/cell-free/txtl_native_real_only_coarsenold.npz", allow_pickle=True)
yT, L, dtT = z["y_seq"], z["lengths"], z["dt_per_sample"]
src_t = np.array([1 if str(s) == "new" else 0 for s in z["source_label"]])
MMd, PMd = 3, 5
uu = z["u_seq"]; OI = [str(c) for c in z["control_names"]].index("u_open")
def uopen_time(i):
    Li = int(L[i]); nz = np.nonzero(uu[i, :Li, OI] > 0)[0]
    return float(np.cumsum(dtT[i, :nz[0] + 1])[-1] / 60.0) if len(nz) else None

# ---------------- right block: prediction cache ----------------
C = np.load(ROOT / "figures/cfps_examples/cache.npz")
pred, ytrue, t_min, Ls, idxs = C["pred"], C["ytrue"], C["t_min"], C["Ls"], C["idxs"]
Mi, Pi = int(C["m_idx"]), int(C["p_idx"])
FAIL = [60, 61, 82, 85]   # 3 good fits + 1 representative failure
def r2(j, ch):
    Lj = Ls[j]; t_, p_ = ytrue[j, :Lj, ch], pred[j, :Lj, ch]
    ss = np.sum((t_ - t_.mean()) ** 2)
    return 1 - np.sum((t_ - p_) ** 2) / ss if ss > 1e-9 else float("nan")

# ======================== figure ========================
fig = plt.figure(figsize=(7.2, 3.4))
# 2 rows shared across both blocks -> bottoms align; thin spacer column between blocks.
gs = fig.add_gridspec(2, 4, wspace=0.40, hspace=0.55,
                      left=0.07, right=0.985, top=0.86, bottom=0.135)
axL = [[fig.add_subplot(gs[r, c]) for c in (0, 1)] for r in (0, 1)]
axR = [[fig.add_subplot(gs[r, c]) for c in (2, 3)] for r in (0, 1)]

# ----- (a) trajectories -----
CH = [("mRNA", MMd, C_MRNA), ("protein", PMd, C_PROT)]
for ri, (sval, glab) in enumerate([(0, "deoxygenated"), (1, "oxygenated")]):
    samp = np.where(src_t == sval)[0]
    tmax = max(np.cumsum(dtT[i, :int(L[i])])[-1] for i in samp) / 60.0
    t_open = float(np.median([x for x in (uopen_time(i) for i in samp) if x is not None])) if sval == 1 else None
    for ci, (cname, ch, color) in enumerate(CH):
        ax = axL[ri][ci]
        for i in samp:
            Li = int(L[i]); t = np.cumsum(dtT[i, :Li]) / 60.0
            ax.plot(t, yT[i, :Li, ch] - yT[i, :Li, ch].min(), color=color, lw=0.3, alpha=0.15)
        ax.set_xlim(0, tmax * 1.02); ax.set_ylim(0, None)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4)); ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        if ri == 1: ax.set_xlabel("time (min)")
        if ci == 0:
            ax.set_ylabel("conc. (nM)")
            ax.set_title(glab, x=-0.34, ha="left", fontsize=8, fontweight="bold", pad=3)
        ax.text(0.05, 0.92, cname, transform=ax.transAxes, va="top", ha="left", fontsize=6.5,
                fontweight="bold", bbox=dict(facecolor=color, alpha=0.18, edgecolor="none", boxstyle="round,pad=0.25"))
        if t_open is not None:
            ax.axvline(t_open, color=UOPEN, ls="--", lw=0.9, alpha=0.9)
            ax.text(t_open, ax.get_ylim()[1] * 0.97, " tube opened", color=UOPEN, fontsize=5.5, va="top", ha="left")

# ----- (b) representative fits -----
for k, j in enumerate(FAIL):
    ax = axR[k // 2][k % 2]
    Lj = Ls[j]; t = t_min[j, :Lj]
    ax.plot(t, ytrue[j, :Lj, Mi], color=C_MRNA, lw=1.0)
    ax.plot(t, pred[j, :Lj, Mi],  color=C_MRNA, lw=1.0, ls="--")
    ax.plot(t, ytrue[j, :Lj, Pi], color=C_PROT, lw=1.0)
    ax.plot(t, pred[j, :Lj, Pi],  color=C_PROT, lw=1.0, ls="--")
    ax.set_xlim(0, t.max()); ax.margins(y=0.08)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4)); ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_title(rf"idx {int(idxs[j])}", fontsize=6)
    if k // 2 == 1: ax.set_xlabel("time (min)")
    if k % 2 == 0: ax.set_ylabel("RFU")

# block titles
fig.text(0.07, 0.915, "(a) measured data", fontsize=8.5, fontweight="bold", ha="left")
fig.text(0.53, 0.915, "(b) representative fits", fontsize=8.5, fontweight="bold", ha="left")

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"cfpe_data_and_fits.{ext}", bbox_inches="tight")
print("wrote", OUT / "cfpe_data_and_fits.pdf")
