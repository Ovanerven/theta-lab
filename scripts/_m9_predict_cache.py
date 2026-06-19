"""Run the M9 event-dark scaffold model (oxy00 s1) on the FULL dataset, cache by global idx.

M9 explicitly models u_open (event-gated maturation -> post-opening protein burst).
We cache predictions for every sample so we can plot the SAME global indices used in the
M4/sLSTM new-bump figure (idx 698, 703).

Output: figures/cfps_examples/cache_m9.npz  (pred, ytrue, t_min, Ls, m_idx, p_idx, is_test)
        index of each row = global dataset idx.
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/Users/olivervanerven/Documents/Thesis/theta-lab"); LLO = ROOT / "last-layer-ode"
sys.path.insert(0, str(LLO))
from plot_diagnostics import (rebuild_model_from_experiment, _test_subset, _maybe_lift,
                              _filter_model_kwargs, load_yaml)
from train import collate, collate_varlen
from metrics.endpoint_r2 import _apply_channel_min_gate as _gate

EXP = ROOT / "experiments_final/FINAL/scaffold_ladder/20260607_140531_M9ed_lr002_oxy00_s1"
OUT = ROOT / "figures" / "cfps_examples"; OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("cpu")

model, ds, *_rest = rebuild_model_from_experiment(EXP, dev); model.eval()
lift_info = _rest[2] or {}; cfg = load_yaml(EXP / "config.yaml")
raw = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds

# which global indices are in M9's test split (for honesty)
tds = _test_subset(ds, EXP)
test_idx = set(tds.indices) if isinstance(tds, torch.utils.data.Subset) else set(range(len(raw)))

# predict on the FULL dataset (rows are in global-idx order)
N = len(raw)
cf = collate_varlen if getattr(raw, "variable_length", False) else collate
b = next(iter(torch.utils.data.DataLoader(raw, batch_size=N, shuffle=False, collate_fn=cf)))
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
pred = pred.cpu().numpy(); ytrue = y.cpu().numpy(); dt = dt.cpu().numpy(); Ls = Ls.cpu().numpy()
m_idx, p_idx = int(oi[0]), int(oi[1])
t_min = np.cumsum(dt, axis=1) / 60.0
is_test = np.array([1 if i in test_idx else 0 for i in range(N)], dtype=np.int8)

print("M9 scaffold_obs_idx (mRNA, protein) =", m_idx, p_idx, "| pred shape", pred.shape)
for gi in (698, 703):
    L = Ls[gi]
    t_, p_ = ytrue[gi, :L, p_idx], pred[gi, :L, p_idx]
    ss = np.sum((t_ - t_.mean())**2); r2 = 1 - np.sum((t_-p_)**2)/ss
    print(f"  global idx {gi}: in_test={bool(is_test[gi])}  protein R2={r2:.3f}")

np.savez(OUT / "cache_m9.npz", pred=pred, ytrue=ytrue, t_min=t_min, Ls=Ls,
         m_idx=m_idx, p_idx=p_idx, is_test=is_test)
print("cached ->", OUT / "cache_m9.npz")
