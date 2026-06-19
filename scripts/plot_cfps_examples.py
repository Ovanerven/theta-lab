"""CFPS prediction-example figures from the best sLSTM model (EZ_slstm_s2, protein R2=0.69).

Plots mRNA (green) and protein (blue) on one axis per test example:
  solid = measured, dashed = predicted.

Modes (argv[1]):
  contact          -- contact sheet of all 125 test examples (to pick from), labelled pos/idx/R2
  contact_good     -- contact sheet of the good band only
  main             -- compact 2x2 main-text figure (4 good examples)
  main_fail        -- compact 2x2 with 3 good + 1 failure
  appendix         -- 4x3 square grid, >=6 good + random rest
  appendix_split   -- two 4x3 square grids split by source (deoxygenated / oxygenated)
  new_bump         -- small 2-panel: sLSTM misses the u_open protein bump (oxygenated)
  compare_m9       -- small 2-panel: sLSTM vs M9-event-dark on the same oxygenated examples
                      (needs cache_m9.npz from scripts/_m9_predict_cache.py)

Output: figures/cfps_examples/*.{pdf,png}
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("/Users/olivervanerven/Documents/Thesis/theta-lab")
OUT = ROOT / "figures" / "cfps_examples"
C = np.load(OUT / "cache.npz")
pred, ytrue, t_min, Ls, idxs = C["pred"], C["ytrue"], C["t_min"], C["Ls"], C["idxs"]
M, P = int(C["m_idx"]), int(C["p_idx"])

# source label (old=deoxygenated, new=oxygenated) per test example
_d = np.load(ROOT / "datasets/cell-free/txtl_native_real_only_coarsenold.npz", allow_pickle=True)
SRC = np.array([str(_d["source_label"][i]) for i in idxs])
SRC_NAME = {"old": "deoxygenated", "new": "oxygenated"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.titlesize": 7.5, "legend.fontsize": 6.5, "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8", "figure.dpi": 300, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "lines.linewidth": 1.1,
})

C_MRNA = "#2C8C3C"   # green
C_PROT = "#3A6FB0"   # blue

def r2(j, ch):
    L = Ls[j]; t_, p_ = ytrue[j, :L, ch], pred[j, :L, ch]
    ss = np.sum((t_ - t_.mean())**2)
    return 1 - np.sum((t_ - p_)**2)/ss if ss > 1e-9 else float("nan")

TWIN = False  # set True for protein (left) / mRNA (right) twin y-axis

def draw(ax, j, title=True, show_xlabel=True, show_ylabel=True):
    L = Ls[j]; t = t_min[j, :L]
    if TWIN:
        ax.plot(t, ytrue[j, :L, P], color=C_PROT, lw=1.1)
        ax.plot(t, pred[j, :L, P],  color=C_PROT, lw=1.1, ls="--")
        axr = ax.twinx()
        axr.plot(t, ytrue[j, :L, M], color=C_MRNA, lw=1.1)
        axr.plot(t, pred[j, :L, M],  color=C_MRNA, lw=1.1, ls="--")
        axr.spines["top"].set_visible(False)
        axr.tick_params(axis="y", colors=C_MRNA, labelsize=6)
        axr.set_ylim(bottom=0)
        ax.tick_params(axis="y", colors=C_PROT)
        if show_ylabel:
            ax.set_ylabel("protein (RFU)", color=C_PROT)
            axr.set_ylabel("mRNA (RFU)", color=C_MRNA)
    else:
        ax.plot(t, ytrue[j, :L, M], color=C_MRNA, lw=1.1)
        ax.plot(t, pred[j, :L, M],  color=C_MRNA, lw=1.1, ls="--")
        ax.plot(t, ytrue[j, :L, P], color=C_PROT, lw=1.1)
        ax.plot(t, pred[j, :L, P],  color=C_PROT, lw=1.1, ls="--")
        if show_ylabel: ax.set_ylabel("RFU")
    ax.set_xlim(0, t.max())
    ax.margins(y=0.05)
    if title:
        ax.set_title(f"test idx {int(idxs[j])}")
    if show_xlabel: ax.set_xlabel("Time (minutes)")

def legend_handles():
    return [
        Line2D([], [], color=C_MRNA, lw=1.3, label="mRNA"),
        Line2D([], [], color=C_PROT, lw=1.3, label="protein"),
        Line2D([], [], color="0.25", lw=1.3, ls="-", label="measured"),
        Line2D([], [], color="0.25", lw=1.3, ls="--", label="predicted"),
    ]

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    print("saved", OUT / f"{name}.png")

# ---------------------------------------------------------------- contact sheet
def contact(positions, name, ncol=5):
    nrow = int(np.ceil(len(positions) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0*ncol, 1.6*nrow), squeeze=False)
    for k, j in enumerate(positions):
        ax = axes[k//ncol][k%ncol]
        draw(ax, j, title=False, show_xlabel=False, show_ylabel=False)
        ax.set_title(f"pos{j} idx{int(idxs[j])}\n$R^2_p$={r2(j,P):.2f} $R^2_m$={r2(j,M):.2f}", fontsize=5.5)
        ax.tick_params(labelsize=4)
    for k in range(len(positions), nrow*ncol):
        axes[k//ncol][k%ncol].axis("off")
    fig.tight_layout()
    save(fig, name)

# ----------------------------------------------------------------- 2x2 figures
def grid22(positions, name, compact=True):
    # compact = half-A4-column width (~3.3in) for text-wrapped placement
    from matplotlib.ticker import MaxNLocator
    figsize = (3.35, 3.0) if compact else (6.5, 4.6)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for k, j in enumerate(positions):
        ax = axes[k//2][k%2]
        draw(ax, j, title=False if compact else True,
             show_xlabel=(k//2 == 1), show_ylabel=(k%2 == 0))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        if compact:
            ax.set_title(f"idx {int(idxs[j])}", fontsize=6)
            ax.tick_params(labelsize=5.5)
            if k % 2 == 0: ax.set_ylabel("RFU", fontsize=6)
            if k // 2 == 1: ax.set_xlabel("Time (min)", fontsize=6)
    fig.legend(handles=legend_handles(), loc="upper center", ncol=4, frameon=False,
               fontsize=5.5, handlelength=1.6, columnspacing=1.0,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.96) if compact else (0, 0, 1, 1))
    save(fig, name)

# ------------------------------------------------ small 2-panel main-body figure
def small2(positions, name, suptitle=None):
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.9))
    for k, j in enumerate(positions):
        ax = axes[k]
        draw(ax, j, title=False, show_xlabel=True, show_ylabel=(k == 0))
        ax.set_title(f"idx {int(idxs[j])}", fontsize=6)
        ax.tick_params(labelsize=5.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xlabel("Time (min)", fontsize=6)
        if k == 0: ax.set_ylabel("RFU", fontsize=6)
    fig.legend(handles=legend_handles(), loc="upper center", ncol=4, frameon=False,
               fontsize=5.5, handlelength=1.6, columnspacing=1.0, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, name)

# --------------------------------------------------------------- appendix grid
def grid_appendix(positions, name, ncol=3, suptitle=None):
    from matplotlib.ticker import MaxNLocator
    nrow = int(np.ceil(len(positions) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5, 6.5), squeeze=False)
    for k, j in enumerate(positions):
        ax = axes[k//ncol][k%ncol]
        draw(ax, j, show_xlabel=(k//ncol == nrow-1), show_ylabel=(k%ncol == 0))
        ax.set_title(f"idx {int(idxs[j])}", fontsize=6.5)
        ax.tick_params(labelsize=5.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    for k in range(len(positions), nrow*ncol):
        axes[k//ncol][k%ncol].axis("off")
    handles = legend_handles()
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    top = 0.965 if suptitle else 1.0
    if suptitle:
        fig.suptitle(suptitle, fontsize=8, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, top))
    save(fig, name)

# ----------------------------- M4(sLSTM) vs M9(event-dark) on the same examples
def compare_m9(global_idxs, name):
    """Per example: measured protein (solid) vs M4 prediction vs M9 prediction (dashed).
    M4 cache is indexed by test position; M9 cache is indexed by global dataset idx."""
    from matplotlib.ticker import MaxNLocator
    M9 = np.load(OUT / "cache_m9.npz")
    p9, y9, t9, L9, P9 = M9["pred"], M9["ytrue"], M9["t_min"], M9["Ls"], int(M9["p_idx"])
    gi2pos = {int(g): k for k, g in enumerate(idxs)}            # global idx -> M4 test pos
    c_meas, c_m4, c_m9 = "0.15", "#E15759", "#3A6FB0"
    fig, axes = plt.subplots(1, len(global_idxs), figsize=(3.35, 1.9), squeeze=False)
    for k, gi in enumerate(global_idxs):
        ax = axes[0][k]
        # measured + M9 from the full-dataset cache (consistent gating/units)
        Lg = L9[gi]; tg = t9[gi, :Lg]
        ax.plot(tg, y9[gi, :Lg, P9], color=c_meas, lw=1.1, label="measured")
        ax.plot(tg, p9[gi, :Lg, P9], color=c_m9, lw=1.1, ls="--", label="M9 (models $u_{open}$)")
        # M4 prediction at the same example
        pos = gi2pos[gi]; Lm = Ls[pos]; tm = t_min[pos, :Lm]
        ax.plot(tm, pred[pos, :Lm, P], color=c_m4, lw=1.1, ls="--", label="M4 (no $u_{open}$)")
        ax.set_title(f"idx {gi}", fontsize=6)
        ax.tick_params(labelsize=5.5)
        ax.set_xlim(0, tg.max()); ax.margins(y=0.05)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xlabel("Time (min)", fontsize=6)
        if k == 0: ax.set_ylabel("protein (RFU)", fontsize=6)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=5.0,
               handlelength=1.6, columnspacing=0.9, bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, name)

def select_source(src, n=12, n_good=6, good_thresh=0.8, seed=0):
    """Pick n positions of a given source: up to n_good good fits + random rest,
    sorted best->worst. Falls back gracefully if not enough good fits exist."""
    rng = np.random.default_rng(seed)
    pool = [j for j in range(len(idxs)) if SRC[j] == src]
    good = [j for j in pool if r2(j, P) > good_thresh]
    ng = min(n_good, len(good))
    pick_good = rng.choice(good, size=ng, replace=False).tolist() if ng else []
    rest = [j for j in pool if j not in pick_good]
    nr = min(n - len(pick_good), len(rest))
    pick_rest = rng.choice(rest, size=nr, replace=False).tolist()
    return sorted(pick_good + pick_rest, key=lambda j: r2(j, P), reverse=True)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "contact"
    # picks chosen after visual review (pos = position in test split)
    # idx: 6, 2, 293, 473  -- all R2_prot>0.97, spanning low->high protein yield
    GOOD = [60, 61, 82, 30]
    # 3 good + 1 representative failure (idx 800: model over-predicts yield, R2=-1.0)
    FAIL = [60, 61, 82, 85]
    # 12 test examples: >=6 good fits + a random draw of the rest (fixed seed),
    # sorted best->worst fit
    rng = np.random.default_rng(0)
    good = [j for j in range(len(idxs)) if r2(j, P) > 0.8]
    pick_good = rng.choice(good, size=6, replace=False).tolist()
    rest = [j for j in range(len(idxs)) if j not in pick_good]
    pick_rest = rng.choice(rest, size=6, replace=False).tolist()
    APPENDIX = sorted(pick_good + pick_rest, key=lambda j: r2(j, P), reverse=True)
    if mode == "contact":
        order = sorted(range(len(idxs)), key=lambda j: r2(j, P), reverse=True)
        contact(order, "contact_all", ncol=7)
    elif mode == "contact_good":
        order = [j for j in sorted(range(len(idxs)), key=lambda j: r2(j, P), reverse=True)
                 if r2(j, P) > 0.4][:35]
        contact(order, "contact_good", ncol=5)
    elif mode == "main":
        grid22(GOOD, "cfps_examples_main")
    elif mode == "main_fail":
        grid22(FAIL, "cfps_examples_main_fail")
    elif mode == "appendix":
        grid_appendix(APPENDIX, "cfps_examples_appendix")
    elif mode == "appendix_split":
        # 9 each (3x3). deoxygenated (old) has many good fits; oxygenated (new) almost none
        old = select_source("old", n=9, n_good=5, good_thresh=0.8)
        new = select_source("new", n=9, n_good=5, good_thresh=0.5)
        grid_appendix(old, "cfps_examples_appendix_deoxygenated",
                      suptitle="Deoxygenated test examples")
        grid_appendix(new, "cfps_examples_appendix_oxygenated",
                      suptitle="Oxygenated test examples")
    elif mode == "new_bump":
        # main-body: model misses the u_open protein bump on oxygenated examples
        # pos 107 = idx 698, pos 109 = idx 703
        small2([107, 109], "cfps_examples_new_bump")
    elif mode == "compare_m9":
        # oxygenated test examples where M9's event mechanism fires (sLSTM never does):
        # idx 709 = M9 tracks the bump; idx 963 = M9 fires a clear (mistimed) bump
        compare_m9([709, 963], "cfps_examples_m9_bump")
    elif mode == "compare_m9_fail":
        # the harder pair where M9 also under-fires (honest counterpoint)
        compare_m9([698, 703], "cfps_examples_m9_bump_fail")
