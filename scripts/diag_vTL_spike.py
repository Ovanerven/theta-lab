"""Counterfactual flux analysis for the v_TL spike hypothesis (M4, sample idx182).

Reloads the trained M4 run, re-runs the forward on the test split, extracts
v_TL(t), M(t), P_imm(t), P_fluor(t) for sample idx182, and computes:
  - total  integral  I = ∫ v_TL·M dt   (= total flux into P_imm)
  - fraction of I that lands in the early "spike window"
  - a counterfactual: clamp v_TL <= cap, re-integrate the ODE, report new P_fluor
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/Users/olivervanerven/Documents/Thesis/theta-lab")
LLO = ROOT / "last-layer-ode"
sys.path.insert(0, str(LLO))

from plot_diagnostics import (
    rebuild_model_from_experiment, _test_subset, _maybe_lift, _filter_model_kwargs,
    load_yaml, device_auto,
)
from train import collate, collate_varlen  # type: ignore
from metrics.endpoint_r2 import _apply_channel_min_gate as _gate

EXP = ROOT / "experiments/FINAL_dense_theta_seeds/20260527_234340_M4_real_s0"
TARGET_IDX = 182

dev = torch.device("cpu")
model, ds, state_names, param_names, lift_info = rebuild_model_from_experiment(EXP, dev)
model.eval()
print("state_names:", state_names)
print("param_names:", param_names)
lift_info = lift_info or {}

cfg = load_yaml(EXP / "config.yaml")

plot_ds = _test_subset(ds, EXP)
raw_ds = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
collate_fn = collate_varlen if getattr(raw_ds, "variable_length", False) else collate
indices = list(plot_ds.indices) if isinstance(plot_ds, torch.utils.data.Subset) else list(range(len(plot_ds)))
print(f"test split size: {len(indices)}; idx{TARGET_IDX} in test:", TARGET_IDX in indices)

loader = torch.utils.data.DataLoader(plot_ds, batch_size=len(indices), shuffle=False, num_workers=0, collate_fn=collate_fn)
batch = next(iter(loader))
y0, u_seq, y_seq, batch_lengths = batch[0], batch[1], batch[2], batch[3]
dt_seq = batch[5] if len(batch) >= 6 else torch.from_numpy(raw_ds.dt[:u_seq.shape[1]])[None, :].expand(y0.shape[0], -1)

if bool(cfg.get("subtract_channel_min", False)):
    cols = cfg.get("subtract_channel_min_cols", None)
    cols = [int(c) for c in cols] if cols is not None else None
    y0, y_seq = _gate(y0, y_seq, cols, batch_lengths)
y0, y_seq = _maybe_lift(y0, y_seq, lift_info or {})

if lift_info:
    obs_idx = torch.tensor(lift_info["scaffold_obs_idx"], dtype=torch.long)
else:
    obs_idx = torch.arange(y0.shape[-1])

mk = {"y_seq": None, "teacher_forcing": False,
      "u_transform": str(cfg.get("u_transform", "none")),
      "y_transform": str(cfg.get("y_transform", "none"))}
with torch.no_grad():
    pred, theta, _beta = model(y0, u_seq, dt_seq, obs_idx, **_filter_model_kwargs(model, mk))

pred = pred.cpu().numpy()       # (B, K, P=4)  full scaffold state
theta = theta.cpu().numpy()     # (B, K, 5)
dt_np = dt_seq.cpu().numpy()
lengths = batch_lengths.cpu().numpy() if batch_lengths is not None else None

pos = indices.index(TARGET_IDX)
Li = int(lengths[pos]) if lengths is not None else pred.shape[1]
t = np.concatenate([[0.0], np.cumsum(dt_np[pos, :Li])])[1:]

M     = pred[pos, :Li, 0]
P_imm = pred[pos, :Li, 1]
P_flu = pred[pos, :Li, 2]
vTX   = theta[pos, :Li, 0]
vTL   = theta[pos, :Li, 1]
k_M   = theta[pos, :Li, 2]
k_mat = theta[pos, :Li, 3]
k_deg = theta[pos, :Li, 4]
dt    = dt_np[pos, :Li]

flux = vTL * M                      # nM/s into P_imm
I_total = float(np.sum(flux * dt))
P_flu_final = float(P_flu[-1])
P_imm_peak = float(P_imm.max())
M_peak = float(M.max())

print("\n================ sample idx182 ================")
print(f"trajectory length Li={Li},  t_end={t[-1]:.0f}s")
print(f"M peak (pred mRNA)      = {M_peak:.2f}")
print(f"P_imm peak              = {P_imm_peak:.2f}")
print(f"P_fluor final (pred pm) = {P_flu_final:.2f}")
print(f"v_TL  min/median/max    = {vTL.min():.4f} / {np.median(vTL):.4f} / {vTL.max():.4f}")
print(f"k_mat min/median/max    = {k_mat.min():.2e} / {np.median(k_mat):.2e} / {k_mat.max():.2e}")
print(f"total  I = ∫v_TL·M dt    = {I_total:.1f}")

# spike window: where v_TL exceeds the user's 0.02 threshold
for cap in (0.02, 0.03):
    mask = vTL > cap
    I_spike = float(np.sum((flux * dt)[mask]))
    t_span = float(np.sum(dt[mask]))
    print(f"\n-- spike defined as v_TL>{cap}: covers {mask.sum()} steps, {t_span:.0f}s "
          f"({100*t_span/t[-1]:.1f}% of time)")
    print(f"   ∫v_TL·M over spike = {I_spike:.1f}  =>  {100*I_spike/I_total:.1f}% of total flux")

# counterfactual: clamp v_TL to 0.02 everywhere, re-integrate the M4 ODE with
# the SAME other thetas and the SAME boluses (use the recorded u-jumps implicitly
# via M, P_imm dynamics).  We re-run only the protein chain driven by the model's
# own M(t) (M is unchanged because v_TL doesn't feed back into M).
from scaffolds import TXTLModel4_ThreeStateScaffold  # noqa

def reintegrate_protein(M_traj, vTL_traj, k_mat_traj, k_deg_traj, dt_traj):
    """Euler re-integration of P_imm, P_fluor given M(t) and rate params.
    dP_imm  = vTL*M - (k_mat+k_deg)*P_imm ;  dP_fluor = k_mat*P_imm."""
    Pi = 0.0; Pf = 0.0
    for k in range(len(M_traj)):
        dPi = vTL_traj[k]*M_traj[k] - (k_mat_traj[k]+k_deg_traj[k])*Pi
        dPf = k_mat_traj[k]*Pi
        Pi = max(0.0, Pi + dPi*dt_traj[k])
        Pf = Pf + dPf*dt_traj[k]
    return Pi, Pf

# sanity: re-integrate with the ORIGINAL v_TL should reproduce P_fluor_final
Pi0, Pf0 = reintegrate_protein(M, vTL, k_mat, k_deg, dt)
print(f"\n[sanity] Euler re-integration with original v_TL -> P_fluor_final={Pf0:.1f} "
      f"(model said {P_flu_final:.1f})")

for cap in (0.02, 0.01):
    vTL_cf = np.minimum(vTL, cap)
    _, Pf_cf = reintegrate_protein(M, vTL_cf, k_mat, k_deg, dt)
    print(f"[counterfactual] clamp v_TL<= {cap}: P_fluor_final={Pf_cf:.1f}  "
          f"({100*Pf_cf/max(Pf0,1e-9):.1f}% of original)")
