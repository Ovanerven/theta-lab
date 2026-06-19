"""SUBRESULT figure (1-column): example protein-fit vs K on the M4 scaffold.

Loads the M4 models at K=1,2,3,6 and dense (seed 0), runs each on ONE representative test
trajectory, and overlays predicted protein P(t) against the real curve. Story: the fit tightens
toward the real trajectory as K grows. No caption (add in LaTeX).

Output: figures/k_ladder/example_fit.{pdf,png}
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/olivervanerven/Documents/Thesis/theta-lab"); LLO = ROOT / "last-layer-ode"
sys.path.insert(0, str(LLO))
from plot_diagnostics import (rebuild_model_from_experiment, _test_subset, _maybe_lift,
                              _filter_model_kwargs, load_yaml)
from train import collate, collate_varlen
from metrics.endpoint_r2 import _apply_channel_min_gate as _gate

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.0, "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
    "figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "lines.linewidth": 1.0,
})

F = ROOT / "experiments_final" / "FINAL"
def find(pat):
    ds = list(F.glob(f"**/*{pat}"));  return ds[0] if ds else None
MODELS = [("$K{=}1$", find("K_M4_k1_s0")), ("$K{=}2$", find("K_M4_k2_s0")),
          ("$K{=}3$", find("K_M4_k3_s0")), ("$K{=}6$", find("K6_M4_s0")),
          ("dense",   find("NG_plain_s0"))]
# distinct but harmonious (Tableau-10 refined; #4E79A7/#F28E2B match the parity figure).
# warm = failing K's (K1,K2), cool = working K's (K3,K6), mauve = dense reference.
COLORS = ["#E15759", "#F28E2B", "#59A14F", "#4E79A7", "#B07AA1"]   # K1, K2, K3, K6, dense
dev = torch.device("cpu")
PM = 2   # M4 scaffold obs index for protein (dataset pm=5 -> scaffold obs [0,2])

def predict(exp):
    model, ds, *_rest = rebuild_model_from_experiment(exp, dev); model.eval()
    lift_info = _rest[2] or {}; cfg = load_yaml(exp / "config.yaml")
    raw = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    pds = _test_subset(ds, exp)
    idxs = list(pds.indices) if isinstance(pds, torch.utils.data.Subset) else list(range(len(pds)))
    cf = collate_varlen if getattr(raw, "variable_length", False) else collate
    b = next(iter(torch.utils.data.DataLoader(pds, batch_size=len(idxs), shuffle=False, collate_fn=cf)))
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
    return pred.cpu().numpy(), y.cpu().numpy(), dt.cpu().numpy(), Ls.cpu().numpy(), np.array(idxs), raw

# predict with all models on the shared split; cache so we pick samples then plot from cache
# args: [TOPN] [mode] [direction]
#   mode = "dense" (rank by dense fit) | "allK" (mean R² across all K)
#   direction = "top" (best-fitting) | "bottom" (worst-fitting — see how each K fails)
TOPN = int(sys.argv[1]) if len(sys.argv) > 1 else 9
MODE = sys.argv[2] if len(sys.argv) > 2 else "dense"
DIRECTION = sys.argv[3] if len(sys.argv) > 3 else "top"
preds = {lab: predict(exp)[0] for lab, exp in MODELS}
dense_pred, ytrue, dtb, Ls, idxs, raw = predict(MODELS[-1][1])
finals = np.array([ytrue[j, Ls[j]-1, PM] for j in range(len(idxs))])

def r2_of(pred, j):
    L = Ls[j]; t_, p_ = ytrue[j, :L, PM], pred[j, :L, PM]
    ss = np.sum((t_ - t_.mean())**2)
    return 1 - np.sum((t_ - p_)**2)/ss if ss > 1e-9 else float("nan")
def traj_r2(j): return r2_of(dense_pred, j)                       # dense-only score
def allK_r2(j): return float(np.mean([r2_of(preds[lab], j) for lab, _ in MODELS]))  # mean over all K
def maxK_r2(j): return float(max(r2_of(preds[lab], j) for lab, _ in MODELS))        # best K achievable
def winner(j):  return max(MODELS, key=lambda m: r2_of(preds[m[0]], j))[0]          # which K fits best
_clean = lambda lab: lab.replace("$", "").replace("{=}", "=")

# rank high-signal samples (final protein above median) by the chosen score
score = {"allK": allK_r2, "maxK": maxK_r2}.get(MODE, traj_r2)
cand = [j for j in range(len(idxs)) if finals[j] > np.median(finals)]
ranked = sorted(cand, key=score, reverse=(DIRECTION == "top"))[:TOPN]
print(f"ranking mode = {MODE} | direction = {DIRECTION}  (n candidates above median yield = {len(cand)})")
if MODE == "maxK":
    from collections import Counter
    tally = Counter(winner(j) for j in ranked)
    print("which K WINS each of the top examples: " +
          "  ".join(f"{_clean(lab)}={tally.get(lab,0)}" for lab, _ in MODELS))
    print("each K's own top-2 best-fit examples (idx:R²):")
    for lab, _ in MODELS:
        t2 = sorted(cand, key=lambda j: r2_of(preds[lab], j), reverse=True)[:2]
        print(f"  {_clean(lab):6s}: " + ", ".join(f"{idxs[j]}:{r2_of(preds[lab], j):.2f}" for j in t2))
print("per-K trajectory R² (look for monotone K1<K2<K3<K6≈dense, no overshoot):")
print("  idx   final   " + "  ".join(f"{lab.replace('$','').replace('{=}','='):>6s}" for lab, _ in MODELS))
for sel in ranked:
    cells = "  ".join(f"{r2_of(preds[lab], sel):6.2f}" for lab, _ in MODELS)
    print(f"  {idxs[sel]:4d}  {finals[sel]:6.0f}   {cells}")

_base = "example_fits" if MODE == "dense" else f"example_fits_{MODE}"
if DIRECTION == "bottom": _base += "_bottom"
out = ROOT / "figures" / "k_ladder" / _base
out.mkdir(parents=True, exist_ok=True)

def _label_right(ax, xr, items):
    """Place color-matched labels at each line's right endpoint, de-overlapped vertically."""
    ax.set_xlim(ax.get_xlim()[0], xr * 1.20)          # room for labels
    y0, y1 = ax.get_ylim(); gap = 0.062 * (y1 - y0)
    items = sorted(items, key=lambda it: it[0])
    ys = [it[0] for it in items]
    for i in range(1, len(ys)):                        # push overlapping labels upward
        if ys[i] - ys[i-1] < gap: ys[i] = ys[i-1] + gap
    if ys[-1] > y1: ax.set_ylim(y0, ys[-1] + gap)
    for (yv, txt, col), y in zip(items, ys):
        ax.text(xr * 1.02, y, txt, color=col, fontsize=6.8, va="center", ha="left",
                fontweight="bold" if txt in ("real", "dense") else "normal")

def plot_one(ax, sel, label_lines=True, title=None):
    t = np.cumsum(dtb[sel, :Ls[sel]]) / 60.0
    items = []
    ax.plot(t, ytrue[sel, :Ls[sel], PM], color="black", lw=1.7, zorder=10)
    items.append((ytrue[sel, Ls[sel]-1, PM], "real", "black"))
    for (lab, exp), c in zip(MODELS, COLORS):
        yv = preds[lab][sel, :Ls[sel], PM]
        ax.plot(t, yv, color=c, lw=1.1)
        items.append((float(yv[-1]), lab, c))          # lab is mathtext e.g. "$K{=}1$"
    ax.set_xlabel("time (min)"); ax.set_ylabel("protein (nM)")
    if label_lines: _label_right(ax, t[-1], items)
    if title: ax.set_title(title, fontsize=6.5)

# individual publication figures (NO title/caption — for direct use in LaTeX)
for rank, sel in enumerate(ranked, 1):
    fig, ax = plt.subplots(figsize=(3.4, 2.7)); plot_one(ax, sel)
    fig.tight_layout()
    tag = f"rank{rank:02d}_idx{idxs[sel]}"
    fig.savefig(out / f"fit_{tag}.pdf", bbox_inches="tight")
    fig.savefig(out / f"fit_{tag}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  rank{rank:02d} idx={idxs[sel]:4d} final={finals[sel]:6.0f}nM dense-R2={traj_r2(sel):.3f}")

# ---- CLEAN paper figure: legend, no grid, distinct colors -> example_fit.{pdf,png}
# optional 4th arg forces a specific dataset idx (e.g. a monotonic-in-K example); else rank-1.
FORCE = int(sys.argv[4]) if len(sys.argv) > 4 else None
sel = int(np.where(idxs == FORCE)[0][0]) if (FORCE is not None and (idxs == FORCE).any()) else ranked[0]
fig, ax = plt.subplots(figsize=(3.4, 2.6))
t = np.cumsum(dtb[sel, :Ls[sel]]) / 60.0
ax.plot(t, ytrue[sel, :Ls[sel], PM], color="black", lw=1.9, label="measured", zorder=10)
for (lab, exp), c in zip(MODELS, COLORS):
    ax.plot(t, preds[lab][sel, :Ls[sel], PM], color=c, lw=1.4, label=lab)
ax.set_xlabel("time (min)"); ax.set_ylabel("protein (nM)"); ax.margins(x=0.01)
ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=6.3,
          handlelength=1.3, columnspacing=1.1, labelspacing=0.3)
fig.tight_layout()
fig.savefig(ROOT / "figures/k_ladder/example_fit.pdf", bbox_inches="tight")
fig.savefig(ROOT / "figures/k_ladder/example_fit.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"wrote clean figures/k_ladder/example_fit.{{pdf,png}} (rank-1 idx={idxs[sel]})")

# contact sheet (titles ON here — for browsing/selection only, not the thesis)
ncol = 3; nrow = int(np.ceil(len(ranked)/ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.0*ncol, 2.3*nrow))
for k, sel in enumerate(ranked):
    ax = axes.flat[k]
    ttl = f"idx {idxs[sel]} | best={_clean(winner(sel))} (R²={maxK_r2(sel):.3f}) | dense={traj_r2(sel):.2f}"
    plot_one(ax, sel, title=ttl)
for k in range(len(ranked), nrow*ncol): axes.flat[k].axis("off")
fig.tight_layout()
fig.savefig(out / "_contactsheet.png", bbox_inches="tight", dpi=150)
print(f"\nwrote {len(ranked)} individual figs + _contactsheet.png in {out}")
