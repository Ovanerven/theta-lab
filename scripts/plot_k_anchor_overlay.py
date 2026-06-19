"""K-anchor trajectory overlay (the 'why' panel for the K-sweep).

Measured protein vs CMVF prediction at K=1 (one constant θ), K=3, and dense (full per-timestep θ),
all M4 / seed 0, on a few representative held-out experiments. Shows WHY too few θ-anchors fail:
a constant θ (K1) cannot bend to the curve; K3 recovers the shape; dense tracks it.

Output: figures/k_ladder/k_anchor_overlay.{pdf,png}
"""
import sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path("/Users/olivervanerven/Documents/Thesis/theta-lab"); sys.path.insert(0, str(ROOT/"last-layer-ode"))
from plot_diagnostics import (rebuild_model_from_experiment, _test_subset, _maybe_lift,
                              _filter_model_kwargs, load_yaml)
from train import collate, collate_varlen
from metrics.endpoint_r2 import _apply_channel_min_gate as _gate

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.3,
})
PM = 2; dev = torch.device("cpu")
F = ROOT/"experiments_final"/"FINAL"
find = lambda pat: next(iter(F.glob(f"**/*{pat}")))
MODELS = [("K=1", find("K_M4_k1_s0"), "#d62728"),
          ("K=2", find("K_M4_k2_s0"), "#ff7f0e"),
          ("K=3", find("K_M4_k3_s0"), "#e7ba52"),
          ("K=6", find("K6_M4_s0"), "#1f77b4"),
          ("dense", find("NG_plain_s0"), "#2ca02c")]

def predict(exp):
    model, ds, *_r = rebuild_model_from_experiment(exp, dev); model.eval()
    li = _r[2] or {}; cfg = load_yaml(exp/"config.yaml")
    raw = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    pds = _test_subset(ds, exp); idxs = list(pds.indices)
    cf = collate_varlen if getattr(raw, "variable_length", False) else collate
    b = next(iter(torch.utils.data.DataLoader(pds, batch_size=len(idxs), shuffle=False, collate_fn=cf)))
    y0, u, y, Ls = b[0], b[1], b[2], b[3]
    dt = b[5] if len(b) >= 6 else torch.from_numpy(raw.dt[:u.shape[1]])[None].expand(y0.shape[0], -1)
    if bool(cfg.get("subtract_channel_min", False)):
        cols = cfg.get("subtract_channel_min_cols"); cols = [int(c) for c in cols] if cols else None
        y0, y = _gate(y0, y, cols, Ls)
    y0, y = _maybe_lift(y0, y, li)
    oi = torch.tensor(li["scaffold_obs_idx"], dtype=torch.long) if li else torch.arange(y0.shape[-1])
    mk = {"y_seq": None, "teacher_forcing": False, "u_transform": str(cfg.get("u_transform", "none")),
          "y_transform": str(cfg.get("y_transform", "none"))}
    with torch.no_grad():
        pred, _, _ = model(y0, u, dt, oi, **_filter_model_kwargs(model, mk))
    return (pred.cpu().numpy(), y.cpu().numpy(), Ls.cpu().numpy(), np.array(idxs), dt.cpu().numpy(), raw)

# predict all models; index each by global sample idx (splits are identical, but align to be safe)
byidx = {}
ref = None
for lab, exp, c in MODELS:
    pred, ytrue, Ls, idxs, dt, raw = predict(exp)
    byidx[lab] = {int(idxs[j]): pred[j, :Ls[j], PM] for j in range(len(idxs))}
    if lab == "dense":
        ref = dict(ytrue=ytrue, Ls=Ls, idxs=idxs, dt=dt, raw=raw,
                   pred={int(idxs[j]): pred[j, :Ls[j], PM] for j in range(len(idxs))})

idxs, Ls, ytrue, dt, raw = ref["idxs"], ref["Ls"], ref["ytrue"], ref["dt"], ref["raw"]
src = np.array([int(raw.source_idx[i]) for i in idxs])         # 0 sealed, 1 opened
finals = np.array([ytrue[j, Ls[j]-1, PM] for j in range(len(idxs))])
def tr2(pred_traj, j):
    L = Ls[j]; t = ytrue[j, :L, PM]; ss = np.sum((t-t.mean())**2)
    return 1 - np.sum((t-pred_traj)**2)/ss if ss > 1e-9 else -np.inf
denseR = np.array([tr2(ref["pred"][int(idxs[j])], j) for j in range(len(idxs))])

# choose 3 representatives: high-signal, dense fits well; prefer a sealed/opened mix
hi = finals > np.percentile(finals, 65)
good = np.where(hi & (denseR > 0.5))[0]
good = good[np.argsort(-finals[good])]
sel = list(good[:2])                                            # 2 highest-signal sealed-ish
opened = [j for j in good if src[j] == 1 and j not in sel]
sel.append(opened[0] if opened else (good[2] if len(good) > 2 else good[0]))
sel = sel[:3]

fig, axes = plt.subplots(1, len(sel), figsize=(2.35*len(sel), 2.2), sharex=False)
if len(sel) == 1: axes = [axes]
for ax, j in zip(axes, sel):
    gi = int(idxs[j]); L = Ls[j]
    t = np.cumsum(dt[j, :L]) / 3600.0                          # hours
    yt = ytrue[j, :L, PM]
    ax.plot(t, yt, "o", color="0.15", ms=2.0, alpha=0.55, label="measured", zorder=5)
    for lab, exp, c in MODELS:
        p = byidx[lab].get(gi)
        if p is not None: ax.plot(t[:len(p)], p, "-", color=c, label=lab)
    ax.set_title(("opened" if src[j] == 1 else "sealed"), fontsize=7)
    ax.set_xlabel("time (h)")
axes[0].set_ylabel("protein (a.u.)")
axes[0].legend(loc="upper left", frameon=False, fontsize=6)
fig.tight_layout()
out = ROOT/"figures"/"k_ladder"
fig.savefig(out/"k_anchor_overlay.pdf", bbox_inches="tight")
fig.savefig(out/"k_anchor_overlay.png", bbox_inches="tight", dpi=300)
print(f"selected test idxs={[int(idxs[j]) for j in sel]} src={[int(src[j]) for j in sel]} "
      f"finals={[round(float(finals[j]),3) for j in sel]}")
print("wrote figures/k_ladder/k_anchor_overlay.{pdf,png}")
