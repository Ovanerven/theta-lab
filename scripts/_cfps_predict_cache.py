"""Run the best sLSTM CFPS model once on the test split, cache predictions + per-example R2.

Output: figures/cfps_examples/cache.npz  (pred, ytrue, t_min, Ls, idxs, m_idx, p_idx)
        figures/cfps_examples/ranking.csv
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

EXP = ROOT / "experiments_final/FINAL/FINAL_coarse_encoder_zoo_light/20260607_180650_EZ_slstm_s2"
OUT = ROOT / "figures" / "cfps_examples"; OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("cpu")

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
    pred, _, _ = model(y0, u, dt, oi, **_filter_model_kwargs(model, mk))
pred = pred.cpu().numpy(); ytrue = y.cpu().numpy(); dt = dt.cpu().numpy(); Ls = Ls.cpu().numpy()
idxs = np.array(idxs)

# scaffold obs indices: [mRNA, protein]
m_idx, p_idx = (int(oi[0]), int(oi[1]))
print("scaffold_obs_idx (mRNA, protein) =", m_idx, p_idx, "| pred shape", pred.shape)

t_min = np.cumsum(dt, axis=1) / 60.0

def r2(j, ch):
    L = Ls[j]; t_, p_ = ytrue[j, :L, ch], pred[j, :L, ch]
    ss = np.sum((t_ - t_.mean())**2)
    return 1 - np.sum((t_ - p_)**2)/ss if ss > 1e-9 else float("nan")

rows = []
for j in range(len(idxs)):
    L = Ls[j]
    rows.append((j, int(idxs[j]), int(L), r2(j, p_idx), r2(j, m_idx),
                 float(ytrue[j, L-1, p_idx]), float(ytrue[j, :L, m_idx].max())))
rows.sort(key=lambda r: r[3], reverse=True)

import csv
with open(OUT / "ranking.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["pos", "idx", "L", "r2_protein", "r2_mrna", "protein_final", "mrna_max"])
    for r in rows: w.writerow([r[0], r[1], r[2], f"{r[3]:.4f}", f"{r[4]:.4f}", f"{r[5]:.2f}", f"{r[6]:.2f}"])

np.savez(OUT / "cache.npz", pred=pred, ytrue=ytrue, t_min=t_min, Ls=Ls, idxs=idxs,
         m_idx=m_idx, p_idx=p_idx)
print(f"cached -> {OUT/'cache.npz'}; ranking -> {OUT/'ranking.csv'}")
print("\nTop 15 by protein R2:")
for r in rows[:15]:
    print(f"  pos={r[0]:3d} idx={r[1]:4d} L={r[2]:3d} r2_prot={r[3]:.3f} r2_mrna={r[4]:.3f} "
          f"p_final={r[5]:.1f} m_max={r[6]:.1f}")
print("\nBottom 8 by protein R2:")
for r in rows[-8:]:
    print(f"  pos={r[0]:3d} idx={r[1]:4d} L={r[2]:3d} r2_prot={r[3]:.3f} r2_mrna={r[4]:.3f} "
          f"p_final={r[5]:.1f} m_max={r[6]:.1f}")
