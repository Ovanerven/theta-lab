"""Overlay all OLD trajectories — original 60s grid vs adaptive-coarsened — to
visually confirm the downsampling preserves trajectory shape (paper-style)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

a = np.load("datasets/cell-free/txtl_native_real_only.npz", allow_pickle=True)
b = np.load("datasets/cell-free/txtl_native_real_only_coarsenold.npz", allow_pickle=True)
src = np.array([str(s) for s in a["source_label"]])
oldi = np.where(src == "old")[0]
MM, PM = 3, 5

fig, ax = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
specs = [("mRNA", MM, "seagreen", (0, 700)), ("Protein", PM, "steelblue", (0, 8500))]
for col, (name, ch, color, ylim) in enumerate(specs):
    for row, (d, tag, nsteps) in enumerate([(a, "ORIGINAL (60s, ~1318 steps)", ""),
                                            (b, "COARSENED (adaptive, ~283 steps)", "")]):
        L = d["lengths"]; dt = d["dt_per_sample"]; y = d["y_seq"]
        axx = ax[row, col]
        for i in oldi:
            Li = int(L[i]); t = np.cumsum(dt[i, :Li]) / 60.0   # minutes
            axx.plot(t, y[i, :Li, ch], color=color, lw=0.4, alpha=0.25)
        axx.set_ylim(*ylim); axx.set_xlim(0, 1500)
        axx.grid(alpha=0.2)
        axx.text(0.03, 0.93, name, transform=axx.transAxes, fontsize=15, fontweight="bold",
                 bbox=dict(facecolor=color, alpha=0.25, edgecolor="none", boxstyle="round"))
        if col == 0:
            axx.set_ylabel("Concentration (nM)", fontsize=12)
        if row == 1:
            axx.set_xlabel("Time (min)", fontsize=12)
        if col == 1:
            axx.text(0.97, 0.5, tag, transform=axx.transAxes, fontsize=11, rotation=270,
                     va="center", ha="left", color="0.3")
ax[0, 0].set_title("ORIGINAL  —  60s grid (~1318 steps)", fontsize=12)
ax[1, 0].set_title("COARSENED  —  adaptive grid (~283 steps, fine→600s tail)", fontsize=12)
fig.suptitle("OLD trajectories: original vs adaptive-coarsened (n=691)", fontsize=14)
fig.tight_layout()
out = "scripts/coarsen_old_compare.png"
fig.savefig(out, dpi=130); print("saved", out)
