"""Learned kinetics, K=6 vs dense (the 'what the anchors do to theta' panel for the K-sweep).

For one representative held-out experiment, plot the five M4 effective parameters theta(t) emitted by
the sparse K=6 model (piecewise-constant) and by the dense model (one value per step).
Shows directly what the anchor budget costs: K=6 holds theta on a few segments, dense bends it freely.

Output: figures/k_ladder/k_theta_compare.{pdf,png}
"""
import sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = Path("/Users/olivervanerven/Documents/Thesis/theta-lab"); sys.path.insert(0, str(ROOT/"last-layer-ode"))
from plot_diagnostics import (rebuild_model_from_experiment, _test_subset, _maybe_lift,
                              _filter_model_kwargs, load_yaml)
from train import collate, collate_varlen
from metrics.endpoint_r2 import _apply_channel_min_gate as _gate

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.titlesize": 8, "legend.fontsize": 7, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "lines.linewidth": 1.4,
})
PM = 2; dev = torch.device("cpu")
F = ROOT/"experiments_final"/"FINAL"
find = lambda pat: next(iter(F.glob(f"**/*{pat}")))

C_K3, C_DENSE = "#d1812b", "#2ca02c"            # K=6 (orange), dense (green)
# M4 theta order from the scaffold forward(): v_TX, v_TL, k_M, k_mat, k_degp
THETA_LABELS = [r"$v_{\mathrm{TX}}$", r"$v_{\mathrm{TL}}$", r"$k_{M}$",
                r"$k_{\mathrm{mat}}$", r"$k_{\mathrm{deg},p}$"]

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
        pred, theta, _ = model(y0, u, dt, oi, **_filter_model_kwargs(model, mk))
    return dict(pred=pred.cpu().numpy(), theta=theta.cpu().numpy(), ytrue=y.cpu().numpy(),
                Ls=Ls.cpu().numpy(), idxs=np.array(idxs), dt=dt.cpu().numpy(), raw=raw)

K3 = predict(find("K6_M4_s0"))
DN = predict(find("NG_plain_s0"))

# align on shared global sample idx
dn_pos = {int(g): j for j, g in enumerate(DN["idxs"])}
shared = [int(g) for g in K3["idxs"] if int(g) in dn_pos]

# Pick a clean deoxygenated, high-signal sample that the dense model fits well.
raw = DN["raw"]; src = {int(g): int(raw.source_idx[int(g)]) for g in shared}
def densefit(g):
    j = dn_pos[g]; L = DN["Ls"][j]; t = DN["ytrue"][j, :L, PM]
    ss = np.sum((t-t.mean())**2)
    return 1 - np.sum((t-DN["pred"][j, :L, PM])**2)/ss if ss > 1e-9 else -np.inf
def final(g):
    j = dn_pos[g]; return DN["ytrue"][j, DN["Ls"][j]-1, PM]
cands = [g for g in shared if src[g] == 0 and densefit(g) > 0.6]
cands.sort(key=final, reverse=True)
gi = cands[0] if cands else max(shared, key=final)

# theta(t) for the chosen sample
jK, jD = list(K3["idxs"]).index(gi), dn_pos[gi]
LK, LD = K3["Ls"][jK], DN["Ls"][jD]
tK = np.cumsum(K3["dt"][jK, :LK]) / 3600.0
tD = np.cumsum(DN["dt"][jD, :LD]) / 3600.0
thK, thD = K3["theta"][jK, :LK], DN["theta"][jD, :LD]

fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.75))
for d, ax in enumerate(axes):
    ax.plot(tD, thD[:, d], color=C_DENSE, label="dense")
    ax.plot(tK, thK[:, d], color=C_K3, label="K=6")
    ax.set_title(THETA_LABELS[d])
    ax.set_xlabel("time (h)")
    ax.set_ylim(bottom=0)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(5.5)
    ax.margins(x=0.02)
axes[0].set_ylabel(r"$\theta(t)$")
fig.legend(handles=[Line2D([], [], color=C_DENSE, lw=1.6, label="dense"),
                    Line2D([], [], color=C_K3, lw=1.6, label="K=6")],
           loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = ROOT/"figures"/"k_ladder"; out.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(out/f"k_theta_compare.{ext}", bbox_inches="tight")
print(f"chosen global idx={gi} (src={src[gi]}, dense protein R2={densefit(gi):.2f})")
print("wrote figures/k_ladder/k_theta_compare.{pdf,png}")
