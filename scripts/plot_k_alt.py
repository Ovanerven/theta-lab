"""Alternative presentations of the K-sweep (cross-sample endpoint R², from r2_cache):
  (A) heatmap  — scaffolds (rows) x K (cols), protein R², annotated  [compact, all scaffolds]
  (B) M4-only  — protein & mRNA R² vs K, dense as dashed ceiling     [clean flagship curve]
Both Helvetica, no captions. Output: figures/k_ladder/k_alt_heatmap.* and k_alt_m4.*
"""
import glob, os, csv
from pathlib import Path
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4, "lines.linewidth": 1.0,
})
ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "experiments_final" / "FINAL"
# M9 omitted: its K-sweep (event_dark) is pending; dense row is event_dark while K-runs are not yet
# regenerated -> mixing would mislead. Add M9 back after FINAL_coarse_K_M9ed lands.
SCAF = {"txtl_model3_two_state": "M3", "txtl_model4_three_state": "M4",
        "txtl_resource_and_maturation_dna": "M5", "txtl_model7_bg_fixed": "M7",
        "txtl_model8_bg_fixed": "M8", "txtl_model9_event_dark": "M9"}
BEST = {"M3": 400, "M4": 300, "M5": 400, "M7": 600, "M8": 600, "M9": 400}
ROWS = ["M3", "M4", "M5", "M7", "M8", "M9"]; XK = [1, 2, 3, 6, "dense"]

data = {s: {k: [] for k in XK} for s in ROWS}          # protein
mdata = {s: {k: [] for k in XK} for s in ROWS}         # mRNA
for cfgp in glob.glob(str(FINAL / "**" / "config.yaml"), recursive=True):
    d = Path(cfgp).parent; cfg = yaml.safe_load(open(cfgp))
    sc = SCAF.get(str(cfg.get("scaffold", "")))
    if not sc: continue
    rc = d / "r2_cache.csv"
    if not (rc.exists() and os.path.getsize(rc) > 50): continue
    r = list(csv.DictReader(open(rc)))[-1]
    pm, mr = float(r["r2_protein_final"]), float(r["r2_mrna_max"])
    K = cfg.get("n_theta_anchors"); en = str(cfg.get("exp_name", ""))
    if K is None:
        if "scaffold_ladder" not in str(cfgp): continue     # dense ONLY from the ladder (no theta_freeze/node_baselines leak)
        if sc == "M4" and "lateP" in en: continue
        if sc == "M9" and "oxy01" in en: continue           # M9 dense = event_dark oxy00 only
        if cfg.get("hidden") == BEST[sc]: data[sc]["dense"].append(pm); mdata[sc]["dense"].append(mr)
    elif int(K) in (1, 2, 3, 6):
        data[sc][int(K)].append(pm); mdata[sc][int(K)].append(mr)

mean = lambda dd, s, k: (np.mean(dd[s][k]) if dd[s][k] else np.nan)

# ---- (A) heatmap: scaffolds x K, protein R² ----
M = np.array([[mean(data, s, k) for k in XK] for s in ROWS])
from matplotlib.colors import TwoSlopeNorm
norm = TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=0.75)   # white/yellow exactly at R²=0
fig, ax = plt.subplots(figsize=(3.4, 2.5))
im = ax.imshow(M, cmap="RdYlGn", norm=norm, aspect="auto")
ax.axvline(3.5, color="white", lw=3.0)                   # separate 'dense' (K=∞ reference)
ax.set_xticks(range(len(XK))); ax.set_xticklabels([str(k) for k in XK])
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels(ROWS)
ax.set_xlabel(r"$\theta$-anchors $K$")
for i in range(len(ROWS)):
    for j in range(len(XK)):
        v = M[i, j]
        if np.isnan(v): ax.text(j, i, "–", ha="center", va="center", fontsize=7, color="0.5"); continue
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.3,
                color="white" if v < -0.05 else "0.1")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(r"$R^2$ (protein)", fontsize=6.5)
cb.ax.tick_params(labelsize=6); ax.grid(False)
fig.tight_layout()
fig.savefig(ROOT / "figures/k_ladder/k_alt_heatmap.pdf", bbox_inches="tight")
fig.savefig(ROOT / "figures/k_ladder/k_alt_heatmap.png", bbox_inches="tight", dpi=300)

# ---- (B) M4-only: protein & mRNA vs K, dense as dashed ceiling ----
xs = list(range(len(XK)))
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for dd, c, lab in [(data, "#1f77b4", "protein"), (mdata, "#d62728", "mRNA max")]:
    ys = [mean(dd, "M4", k) for k in XK]
    ax.plot(xs[:-1], ys[:-1], "-o", color=c, label=lab, ms=4)          # K=1..6
    ax.axhline(ys[-1], color=c, ls="--", lw=1.0, alpha=0.8)            # dense ceiling
    ax.text(xs[-2] + 0.05, ys[-1], f" dense", color=c, fontsize=6.3, va="bottom")
ax.set_xticks(xs[:-1]); ax.set_xticklabels([str(k) for k in XK[:-1]])
ax.set_xlabel(r"$\theta$-anchors $K$"); ax.set_ylabel(r"$R^2$  (M4)")
ax.legend(loc="lower right", frameon=True)
fig.tight_layout()
fig.savefig(ROOT / "figures/k_ladder/k_alt_m4.pdf", bbox_inches="tight")
fig.savefig(ROOT / "figures/k_ladder/k_alt_m4.png", bbox_inches="tight", dpi=300)
print("wrote k_alt_heatmap.{pdf,png} and k_alt_m4.{pdf,png}")
