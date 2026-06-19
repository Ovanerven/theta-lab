"""Summary figure of the real cell-free data: a couple of example experiments, each shown as a row of
input schedule -> mRNA -> protein. The input panel is a colored bar chart of the reagent additions
u(t) (which species, when, how much), with a shared legend. Pure data figure (no model).

Output: figures/cfpe_data_summary/cfpe_data_summary.{pdf,png}
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch

ROOT = Path("/Users/olivervanerven/Documents/Thesis/theta-lab")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "legend.title_fontsize": 9.5,
    "figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
})
MRNA_C, PROT_C = "#2e8b57", "#4e79a7"
MM, PM = 3, 5   # dataset obs cols: mRNA, protein

z = np.load(ROOT / "datasets/cell-free/txtl_native_real_only_coarsenold.npz", allow_pickle=True)
y = z["y_seq"]; u = z["u_seq"]; L = z["lengths"]; dt = z["dt_per_sample"]
src = np.array([str(s) for s in z["source_label"]])             # 'old'=deoxygenated, 'new'=oxygenated
cn = [str(c) for c in z["control_names"]]
OI = cn.index("u_open")
REAG = [j for j in range(u.shape[-1]) if j != OI]               # reagent channels (drop the open event)
SHORT = {"DNA c": "DNA", "Lysate 2%PEG": "Lysate"}             # tidy display names
RNAME = {j: SHORT.get(cn[j], cn[j]) for j in REAG}
PALETTE = ["#4e79a7", "#59a14f", "#9c9ede", "#76b7b2", "#54a24b", "#a0cbe8",
           "#b07aa1", "#f28e2b", "#e15759", "#bab0ac", "#86bcb6", "#d37295"]
RCOLOR = {j: PALETTE[k % len(PALETTE)] for k, j in enumerate(REAG)}

def mins(i):
    Li = int(L[i]); return np.cumsum(dt[i, :Li]) / 60.0
def hrs(i):
    Li = int(L[i]); return np.cumsum(dt[i, :Li]) / 3600.0
def gated(i, ch):
    Li = int(L[i]); seg = y[i, :Li, ch]; return seg - seg.min()
def endpoint_protein(i):
    return gated(i, PM)[-1]

ep = np.array([endpoint_protein(i) for i in range(len(y))])
old = np.where(src == "old")[0]
def pick(pool, q): return pool[np.argsort(ep[pool])][int(q * (len(pool) - 1))]
EXAMPLES = [pick(old, 0.90), pick(old, 0.30)]                  # a high- and a low-yield deoxygenated run

# common scales across rows
feed_max_min = max((mins(i)[np.nonzero(u[i, :int(L[i])][:, REAG].sum(1) > 0)[0]].max()) for i in EXAMPLES)
vol_max = max((u[i, :int(L[i])][:, REAG].max()) for i in EXAMPLES)
present = [j for j in REAG if any(u[i, :int(L[i]), j].max() > 0 for i in EXAMPLES)]
# shared y-limit per species (across rows) so the difference in final outcome between the two
# example runs is visible: mRNA panels share one scale, protein panels share another.
mrna_top = max(gated(i, MM).max() for i in EXAMPLES)
prot_top = max(gated(i, PM).max() for i in EXAMPLES)

n = len(EXAMPLES)
fig = plt.figure(figsize=(6.7, 1.02 * n + 0.5))
# columns: [inputs | gap (arrow) | mRNA | protein]; left margin holds the legend
gs = fig.add_gridspec(n, 4, width_ratios=[0.86, 0.18, 0.62, 0.62],
                      left=0.215, right=0.98, top=0.82, bottom=0.21, wspace=0.30, hspace=0.55)
ax_in = [fig.add_subplot(gs[r, 0]) for r in range(n)]
ax_mr = [fig.add_subplot(gs[r, 2]) for r in range(n)]
ax_pr = [fig.add_subplot(gs[r, 3]) for r in range(n)]

for ri, i in enumerate(EXAMPLES):
    Li = int(L[i]); tm = mins(i); th = hrs(i)
    # --- inputs: colored bar chart of reagent additions (log volume) ---
    ax = ax_in[ri]
    bars = []   # (time, volume, color); drawn tall-first so small bars stay visible
    for j in REAG:
        for k in np.nonzero(u[i, :Li, j] > 0)[0]:
            bars.append((tm[k], float(u[i, k, j]), RCOLOR[j]))
    for t0, h, c in sorted(bars, key=lambda b: -b[1]):
        ax.bar(t0, h, width=1.6, color=c, edgecolor="0.25", linewidth=0.2, zorder=2)
    ax.set_yscale("log"); ax.set_ylim(10, vol_max * 1.6); ax.set_xlim(-2, feed_max_min * 1.06)
    ax.set_ylabel("volume (µL)")
    # --- mRNA and protein trajectories ---
    for ax2, ch, col, top in [(ax_mr[ri], MM, MRNA_C, mrna_top), (ax_pr[ri], PM, PROT_C, prot_top)]:
        ax2.plot(th, gated(i, ch), color=col, lw=1.2)
        ax2.set_ylim(0, top * 1.05); ax2.set_xlim(0, th[-1] * 1.02)
    ax_mr[ri].set_ylabel("conc. (nM)")   # unit on the mRNA panels only
    if ri == n - 1:
        ax_in[ri].set_xlabel("time (min)")
        ax_mr[ri].set_xlabel("time (h)"); ax_pr[ri].set_xlabel("time (h)")

# titles / group headers
# boxed in-panel channel labels (same style as figures/cfpe_split/cfpe_trajectories) instead of
# colored titles: keeps the visual language consistent and frees the title band. On every row.
for r in range(n):
    for ax2, lab, col in [(ax_mr[r], "mRNA", MRNA_C), (ax_pr[r], "Protein", PROT_C)]:
        ax2.text(0.05, 0.92, lab, transform=ax2.transAxes, va="top", ha="left", fontsize=9,
                 fontweight="bold", bbox=dict(facecolor=col, alpha=0.18, edgecolor="none",
                                              boxstyle="round,pad=0.25"))
p_in, p_mr, p_pr = ax_in[0].get_position(), ax_mr[0].get_position(), ax_pr[0].get_position()
xc_in = 0.5 * (p_in.x0 + p_in.x1)
xc_out = 0.5 * (p_mr.x0 + p_pr.x1)
fig.text(xc_in, 0.93, "Inputs $u(t)$", ha="center", va="center", fontsize=12, fontweight="bold")
fig.text(xc_out, 0.93, "Observed species $y$", ha="center", va="center", fontsize=12, fontweight="bold")
# panel letters to the left of each group title (not at the bottom)
fig.text(p_in.x0, 0.93, "(a)", ha="left", va="center", fontsize=11, fontweight="bold")
fig.text(p_mr.x0, 0.93, "(b)", ha="left", va="center", fontsize=11, fontweight="bold")

# one arrow per row: this recipe -> that output
for r in range(n):
    pr_in, pr_mr = ax_in[r].get_position(), ax_mr[r].get_position()
    yc = 0.5 * (pr_in.y0 + pr_in.y1)
    x_end = pr_mr.x0 - 0.058                        # stop clear of the "conc. (nM)" label
    x_start = max(pr_in.x1 + 0.010, x_end - 0.062)  # short visible stem before the head
    fig.add_artist(FancyArrowPatch((x_start, yc), (x_end, yc),
                                   transform=fig.transFigure,
                                   arrowstyle="-|>,head_length=0.5,head_width=0.22",
                                   mutation_scale=9, lw=0.9, color="0.3", joinstyle="miter"))

# shared reagent legend in the left margin, next to the bar plots
handles = [Patch(facecolor=RCOLOR[j], edgecolor="0.25", linewidth=0.2, label=RNAME[j]) for j in present]
fig.legend(handles=handles, title="reagent", loc="center left", bbox_to_anchor=(0.005, 0.5),
           frameon=False, handlelength=0.9, handleheight=0.9, labelspacing=0.3, borderaxespad=0.0)
out = ROOT / "figures" / "cfpe_data_summary"; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "cfpe_data_summary.pdf", bbox_inches="tight")
fig.savefig(out / "cfpe_data_summary.png", bbox_inches="tight", dpi=300)
print("wrote", out / "cfpe_data_summary.{pdf,png}")
print("examples (idx, endpoint protein):", [(int(i), round(float(ep[i]), 1)) for i in EXAMPLES])
