"""theta(t) over time for the best sLSTM M4 model (EZ_slstm_s2, protein R2=0.69) on the test split.

One panel per learned kinetic parameter, every test trajectory overlaid (faint), x = minutes.
Mirrors scripts/_cfps_predict_cache.py for model loading; the model forward returns
(y_pred, theta, beta), so theta(t) is the second output.

Output: figures/cfps_examples/theta_over_time.{pdf,png}
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

EXP = ROOT / "experiments_final/FINAL/FINAL_coarse_encoder_zoo_light/20260607_180650_EZ_slstm_s2"
OUT = ROOT / "figures" / "cfps_examples"; OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("cpu")

# M4 (txtl_model4_three_state) theta order: v_TX, v_TL, k_M, k_mat, k_degp
THETA_LABELS = [r"$v_{TX}$", r"$v_{TL}$", r"$k_M$", r"$k_{mat}$", r"$k_{deg,p}$"]

model, ds, *_rest = rebuild_model_from_experiment(EXP, dev); model.eval()
lift_info = _rest[2] or {}; cfg = load_yaml(EXP / "config.yaml")
raw = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
pds = _test_subset(ds, EXP)
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
    pred, theta, _ = model(y0, u, dt, oi, **_filter_model_kwargs(model, mk))

theta = theta.cpu().numpy()            # (B, K, theta_dim)
Ls = Ls.cpu().numpy()
t_min = np.cumsum(dt.cpu().numpy(), axis=1) / 60.0
B, K, D = theta.shape
print(f"theta shape {theta.shape}; labels {len(THETA_LABELS)}")
assert D == len(THETA_LABELS), f"theta_dim {D} != {len(THETA_LABELS)} labels"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})
rng = np.random.default_rng(0)
colors = plt.cm.tab20(rng.uniform(0, 1, B))

fig, axes = plt.subplots(D, 1, figsize=(6.0, 1.6 * D), sharex=True)
for d, ax in enumerate(axes):
    for j in range(B):
        L = int(Ls[j])
        ax.plot(t_min[j, :L], theta[j, :L, d], color=colors[j], lw=0.6, alpha=0.5)
    ax.set_ylabel(THETA_LABELS[d])
    ax.set_ylim(bottom=0)
    ax.margins(x=0)
axes[-1].set_xlabel("Time (minutes)")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"theta_over_time.{ext}", bbox_inches="tight")
print("saved", OUT / "theta_over_time.png")
