"""Parameter-gating adaptation figure: one sample, three M4 models.
Rows: (a) M4 (all theta free), (b) v_TX frozen (retrain), (c) k_mat frozen (retrain).
Cols: theta(t) | mRNA fit | protein fit. Shows how the FREE parameters adapt when one is pinned.

Output: figures/freeze_adapt.{pdf,png}
Usage:  python scripts/plot_freeze_adapt.py [sample_idx]
"""
import sys, glob
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path("/Users/olivervanerven/Documents/Thesis/theta-lab"); LLO = ROOT / "last-layer-ode"
sys.path.insert(0, str(LLO))
from plot_diagnostics import (rebuild_model_from_experiment, _maybe_lift, _filter_model_kwargs, load_yaml)
from train import collate, collate_varlen
from metrics.endpoint_r2 import _apply_channel_min_gate as _gate

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.2, "figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
})
F = ROOT / "experiments_final/FINAL"
def find(pat, base="theta_freeze"):
    ds = sorted((F / base).glob(f"*{pat}*"));  return ds[0]
MODELS = [("(a) M4, all free",        find("NG_plain_s0", "scaffold_ladder"), None),
          ("(b) $v_{TX}$ frozen",     find("FZ_vTX_s0"),  0),
          ("(c) $k_{mat}$ frozen",    find("FZ_kmat_s0"), 3)]
# theta order (M4): v_TX, v_TL, k_M, k_mat, k_degp
TH_NAMES = [r"$v_{TX}$", r"$v_{TL}$", r"$k_M$", r"$k_{mat}$", r"$k_{degp}$"]
TH_COL   = ["#E15759", "#F28E2B", "#59A14F", "#4E79A7", "#B07AA1"]
MM_C, PM_C = "#2e8b57", "#4e79a7"   # mRNA green, protein blue
dev = torch.device("cpu")

def load(exp):
    model, ds, *_r = rebuild_model_from_experiment(exp, dev); model.eval()
    lift = _r[2] or {}; cfg = load_yaml(exp / "config.yaml")
    raw = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    return model, raw, lift, cfg, np.load(exp / "split.npz")

def run(model, raw, lift, cfg, idx):
    sub = torch.utils.data.Subset(raw, [idx])
    cf = collate_varlen if getattr(raw, "variable_length", False) else collate
    b = next(iter(torch.utils.data.DataLoader(sub, batch_size=1, shuffle=False, collate_fn=cf)))
    y0, u, y, Ls = b[0], b[1], b[2], b[3]
    dt = b[5] if len(b) >= 6 else torch.from_numpy(raw.dt[:u.shape[1]])[None].expand(y0.shape[0], -1)
    if bool(cfg.get("subtract_channel_min", False)):
        cols = cfg.get("subtract_channel_min_cols"); cols = [int(c) for c in cols] if cols else None
        y0, y = _gate(y0, y, cols, Ls)
    y0, y = _maybe_lift(y0, y, lift)
    oi = torch.tensor(lift["scaffold_obs_idx"], dtype=torch.long) if lift else torch.arange(y0.shape[-1])
    mk = {"y_seq": None, "teacher_forcing": False, "u_transform": str(cfg.get("u_transform", "none")),
          "y_transform": str(cfg.get("y_transform", "none"))}
    with torch.no_grad():
        pred, theta, _ = model(y0, u, dt, oi, **_filter_model_kwargs(model, mk))
    L = int(Ls[0]); t = np.cumsum(dt[0, :L].numpy()) / 60.0
    pred, theta, y = pred[0].numpy(), theta[0].numpy(), y[0].numpy()
    return dict(t=t, th=theta[:L, :5], predM=pred[:L, 0], predP=pred[:L, 2],
                measM=y[:L, 0], measP=y[:L, 2])

# ---- pick a well-fit, high-signal test sample using the normal model ----
m0, raw0, lift0, cfg0, split0 = load(MODELS[0][1])
def pr2(d):
    t, p = d["measP"], d["predP"]; ss = np.sum((t - t.mean())**2)
    return 1 - np.sum((t - p)**2)/ss if ss > 1e-9 else -9
if len(sys.argv) > 1:
    SAMPLE = int(sys.argv[1])
else:
    test = list(split0["test_idx"])
    cand = sorted(test, key=lambda i: pr2(run(m0, raw0, lift0, cfg0, i)), reverse=True)
    SAMPLE = cand[0]   # best-fit test sample under the normal model
print("sample idx:", SAMPLE)

# ---- gather all three models on that sample ----
DATA = []
for label, exp, frozen in MODELS:
    model, raw, lift, cfg, _ = load(exp)
    DATA.append((label, frozen, run(model, raw, lift, cfg, SAMPLE)))

# ---- plot: 3 rows x 3 cols ----
fig = plt.figure(figsize=(8.2, 6.0))
gs = GridSpec(3, 3, figure=fig, width_ratios=[1.15, 1.0, 1.0], wspace=0.34, hspace=0.42)
for ri, (label, frozen, d) in enumerate(DATA):
    # theta(t)
    axT = fig.add_subplot(gs[ri, 0])
    for j in range(5):
        is_fz = (frozen == j)
        axT.plot(d["t"], d["th"][:, j], color=TH_COL[j], lw=(1.8 if is_fz else 1.1),
                 ls=("--" if is_fz else "-"), alpha=(1.0 if is_fz else 0.9),
                 label=TH_NAMES[j] + (" (frozen)" if is_fz else ""))
    axT.set_yscale("log"); axT.set_xlabel("time (min)"); axT.set_ylabel(r"$\theta(t)$")
    axT.set_title(label, fontsize=9, loc="left", fontweight="bold")
    if frozen is not None:   # mark the pinned parameter on its (flat, dashed) curve
        axT.text(d["t"][-1], d["th"][-1, frozen], " frozen", color=TH_COL[frozen],
                 fontsize=6, va="center", ha="left", fontweight="bold")
    # mRNA + protein fits
    for ci, (ch, col, ml, pl, ylab) in enumerate([
        ("M", MM_C, d["measM"], d["predM"], "mRNA (nM)"),
        ("P", PM_C, d["measP"], d["predP"], "protein (nM)")]):
        ax = fig.add_subplot(gs[ri, ci + 1])
        ax.plot(d["t"], ml, color="black", lw=1.6, label="measured", zorder=3)
        ax.plot(d["t"], pl, color=col, lw=1.4, ls="--", label="predicted", zorder=2)
        ax.set_xlabel("time (min)"); ax.set_ylabel(ylab); ax.set_ylim(0, None)
        if ri == 0 and ci == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=6.2, handlelength=1.4)

# shared parameter legend across the top
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=TH_COL[j], lw=1.6, label=TH_NAMES[j]) for j in range(5)]
fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=7.5,
           bbox_to_anchor=(0.5, 1.03), handlelength=1.6, columnspacing=1.6)

out = ROOT / "figures" / "freeze_adapt"
fig.savefig(str(out) + ".pdf", bbox_inches="tight")
fig.savefig(str(out) + ".png", bbox_inches="tight", dpi=300)
print(f"wrote figures/freeze_adapt.{{pdf,png}}  (sample {SAMPLE})")
