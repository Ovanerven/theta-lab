"""General failure-case diagnostics for theta-lab ODE-RNN runs.

Loads a trained run, re-runs the forward on the held-out TEST split, and
quantifies *why* the endpoint predictions fail — several independent ways, so
no single eyeballed hypothesis (e.g. "the v_TL spike") gets to drive decisions
unchecked.

Reports (printed + CSV + plots into <exp>/diagnostics/):
  1. Per-sample endpoint table  (mm-max, pm-final: true vs pred, signed/log err,
     source old/new, input-OOD distance).
  2. Bias-vs-variance:  mean signed error overall, by source, by true-magnitude
     tercile  → is the model systematically over/under, and where?
  3. Failure ranking:  worst over- and under-predictors for pm and mm.
  4. theta bound-saturation:  for each theta dim, % of timesteps pinned at its
     lo/hi bound (encoder hitting the rails), and whether saturation tracks error.
  5. theta–error correlation:  per-sample time-mean of each theta vs signed
     pm-error (rank corr) → which kinetic knob co-moves with the failure.
  6. Input-OOD:  does endpoint error grow with distance from the train input
     distribution?  (failure = bad generalization vs failure = model capacity)
  7. Scaffold-specific flux attribution (M4): protein endpoint = integral of
     v_TL*M; report the integral, maturation fraction, and the M-integral error.

Usage:
  python scripts/diag_failures.py [EXP_DIR]
  (defaults to the M4 dense-seed run)
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
    load_yaml,
)
from train import collate, collate_varlen  # type: ignore
from metrics.endpoint_r2 import _apply_channel_min_gate as _gate
from scaffolds import SCAFFOLDS

EXP = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / "experiments/FINAL_dense_theta_seeds/20260527_234340_M4_real_s0")


# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────
def rank_corr(a, b):
    """Spearman rank correlation without scipy."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    ra = np.argsort(np.argsort(a[ok])); rb = np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def r2(true, pred):
    true = np.asarray(true, float); pred = np.asarray(pred, float)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ────────────────────────────────────────────────────────────────────────────
# load + forward
# ────────────────────────────────────────────────────────────────────────────
dev = torch.device("cpu")
model, ds, state_names, param_names, lift_info = rebuild_model_from_experiment(EXP, dev)
model.eval()
lift_info = lift_info or {}
cfg = load_yaml(EXP / "config.yaml")
scaffold = SCAFFOLDS[str(cfg["scaffold"])]
theta_lo = np.array(getattr(scaffold, "theta_lo_vec"), float)
theta_hi = np.array(getattr(scaffold, "theta_hi_vec"), float)
obs_state_idx = list(scaffold.obs_state_idx)         # e.g. [0,2] = [M, P_fluor]
MM_STATE, PM_STATE = obs_state_idx[0], obs_state_idx[1]

raw_ds = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
plot_ds = _test_subset(ds, EXP)
indices = list(plot_ds.indices) if isinstance(plot_ds, torch.utils.data.Subset) else list(range(len(plot_ds)))
collate_fn = collate_varlen if getattr(raw_ds, "variable_length", False) else collate

src_all = getattr(raw_ds, "source_idx", None)
src_test = np.array([int(src_all[i]) if src_all is not None else -1 for i in indices])

loader = torch.utils.data.DataLoader(plot_ds, batch_size=len(indices), shuffle=False, num_workers=0, collate_fn=collate_fn)
batch = next(iter(loader))
y0, u_seq, y_seq, batch_lengths = batch[0], batch[1], batch[2], batch[3]
dt_seq = batch[5] if len(batch) >= 6 else torch.from_numpy(raw_ds.dt[:u_seq.shape[1]])[None, :].expand(y0.shape[0], -1)
u_raw = u_seq.clone()  # keep pre-lift controls for OOD features

if bool(cfg.get("subtract_channel_min", False)):
    cols = cfg.get("subtract_channel_min_cols", None)
    cols = [int(c) for c in cols] if cols is not None else None
    y0, y_seq = _gate(y0, y_seq, cols, batch_lengths)
y0, y_seq = _maybe_lift(y0, y_seq, lift_info or {})

obs_idx = (torch.tensor(lift_info["scaffold_obs_idx"], dtype=torch.long)
           if lift_info else torch.arange(y0.shape[-1]))
mk = {"y_seq": None, "teacher_forcing": False,
      "u_transform": str(cfg.get("u_transform", "none")),
      "y_transform": str(cfg.get("y_transform", "none"))}
with torch.no_grad():
    pred, theta, _ = model(y0, u_seq, dt_seq, obs_idx, **_filter_model_kwargs(model, mk))

pred = pred.cpu().numpy()        # (B,K,P) full scaffold state
theta = theta.cpu().numpy()      # (B,K,theta_dim)
y_true = y_seq.cpu().numpy()     # (B,K,P) scaffold space (obs cols populated)
dt_np = dt_seq.cpu().numpy()
u_np = u_raw.cpu().numpy()
lengths = batch_lengths.cpu().numpy() if batch_lengths is not None else np.full(len(indices), pred.shape[1])
B = len(indices)

# ────────────────────────────────────────────────────────────────────────────
# per-sample endpoints + theta stats
# ────────────────────────────────────────────────────────────────────────────
rows = []
theta_mean = np.zeros((B, theta.shape[-1]))
theta_satlo = np.zeros((B, theta.shape[-1]))
theta_sathi = np.zeros((B, theta.shape[-1]))
for j in range(B):
    Li = int(lengths[j])
    mm_t = float(np.max(y_true[j, :Li, MM_STATE]))
    mm_p = float(np.max(pred[j, :Li, MM_STATE]))
    pm_t = float(y_true[j, Li - 1, PM_STATE])
    pm_p = float(pred[j, Li - 1, PM_STATE])
    th = theta[j, :Li, :]
    theta_mean[j] = th.mean(0)
    # within 1% of bound (log-relative) counts as "pinned"
    theta_satlo[j] = np.mean(th <= theta_lo * 1.01, axis=0)
    theta_sathi[j] = np.mean(th >= theta_hi * 0.99, axis=0)
    rows.append(dict(
        pos=j, idx=indices[j], source=("new" if src_test[j] == 1 else "old" if src_test[j] == 0 else "?"),
        mm_true=mm_t, mm_pred=mm_p, mm_err=mm_p - mm_t,
        pm_true=pm_t, pm_pred=pm_p, pm_err=pm_p - pm_t,
        pm_logerr=np.log1p(max(pm_p, 0)) - np.log1p(max(pm_t, 0)),
    ))

import csv
diag_dir = EXP / "diagnostics"; diag_dir.mkdir(exist_ok=True)
with open(diag_dir / "endpoint_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

pm_true = np.array([r["pm_true"] for r in rows]); pm_pred = np.array([r["pm_pred"] for r in rows])
mm_true = np.array([r["mm_true"] for r in rows]); mm_pred = np.array([r["mm_pred"] for r in rows])
pm_err = pm_pred - pm_true; mm_err = mm_pred - mm_true

print("=" * 78)
print(f"FAILURE DIAGNOSTICS  —  {EXP.name}")
print(f"scaffold={cfg['scaffold']}  hidden={cfg.get('hidden')}  test n={B}  "
      f"(old={int((src_test==0).sum())} new={int((src_test==1).sum())})")
print("=" * 78)

# ── 1. global sanity ──
print(f"\n[1] ENDPOINT R²   pm-final={r2(pm_true,pm_pred):.3f}   mm-max={r2(mm_true,mm_pred):.3f}")

# ── 2. bias vs variance ──
print("\n[2] SIGNED-ERROR BIAS (pred-true). +ve = over-predict")
def bias_block(mask, label):
    if mask.sum() == 0: return
    e = pm_err[mask]
    print(f"   pm  {label:16s} n={mask.sum():3d}  mean={e.mean():+8.1f}  median={np.median(e):+8.1f}  "
          f"std={e.std():7.1f}  |over|/n={np.mean(e>0):.2f}")
bias_block(np.ones(B, bool), "ALL")
bias_block(src_test == 0, "old")
bias_block(src_test == 1, "new")
# magnitude terciles on true pm
order = np.argsort(pm_true); terc = np.array_split(order, 3)
for ti, idxs in enumerate(terc):
    m = np.zeros(B, bool); m[idxs] = True
    rng = f"[{pm_true[idxs].min():.0f},{pm_true[idxs].max():.0f}]"
    bias_block(m, f"pm-tercile{ti} {rng}")

# ── 3. failure ranking ──
print("\n[3] WORST pm FAILURES")
si = np.argsort(pm_err)
def show(j):
    r = rows[j]; print(f"   idx{r['idx']:<4d} [{r['source']}]  pm true={r['pm_true']:7.0f} pred={r['pm_pred']:7.0f}  "
                        f"err={r['pm_err']:+7.0f}   mm true={r['mm_true']:6.1f} pred={r['mm_pred']:6.1f}")
print("  -- biggest OVER-predict --");  [show(j) for j in si[::-1][:6]]
print("  -- biggest UNDER-predict --"); [show(j) for j in si[:6]]

# ── 4. theta bound saturation ──
print("\n[4] θ BOUND SATURATION (frac of timesteps pinned, test-mean) + corr(sat,|pm_err|)")
for d in range(theta.shape[-1]):
    nm = param_names[d] if d < len(param_names) else f"θ{d}"
    lo_f = theta_satlo[:, d].mean(); hi_f = theta_sathi[:, d].mean()
    c = rank_corr(theta_satlo[:, d] + theta_sathi[:, d], np.abs(pm_err))
    flag = "  <<" if (lo_f > 0.3 or hi_f > 0.3) else ""
    print(f"   {nm:6s}  lo-pinned={lo_f:.2f}  hi-pinned={hi_f:.2f}   corr(sat,|err|)={c:+.2f}{flag}")

# ── 5. theta–error correlation ──
print("\n[5] rank-corr( time-mean θ , signed pm_err )  → knob that co-moves with overshoot")
cors = []
for d in range(theta.shape[-1]):
    nm = param_names[d] if d < len(param_names) else f"θ{d}"
    c = rank_corr(theta_mean[:, d], pm_err); cors.append((abs(c), nm, c))
for _, nm, c in sorted(cors, reverse=True):
    print(f"   {nm:6s}  ρ={c:+.2f}")

# ── 6. input-OOD ──
split = np.load(EXP / "split.npz")
train_idx = split["train_idx"].tolist() if "train_idx" in split.files else []
if train_idx:
    feat_all = u_np.sum(axis=1)  # (B_test, C) total bolus dose per control (test batch already)
    # build train features by loading train rows through the same dataset
    tr_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(raw_ds, train_idx), batch_size=len(train_idx),
        shuffle=False, num_workers=0, collate_fn=collate_fn)
    tb = next(iter(tr_loader)); u_tr = tb[1].cpu().numpy().sum(axis=1)
    mu = u_tr.mean(0); sd = u_tr.std(0) + 1e-9
    ood = np.sqrt(np.sum(((feat_all - mu) / sd) ** 2, axis=1))  # diag-Mahalanobis
    print(f"\n[6] INPUT-OOD distance vs |pm_err|:  rank-corr={rank_corr(ood,np.abs(pm_err)):+.2f}")
    hi_ood = ood > np.percentile(ood, 75)
    print(f"   |pm_err| mean: OOD-top-quartile={np.abs(pm_err)[hi_ood].mean():.0f}  "
          f"rest={np.abs(pm_err)[~hi_ood].mean():.0f}")
else:
    ood = np.full(B, np.nan)
    print("\n[6] INPUT-OOD: no train_idx in split.npz; skipped")

# ── 7. M4 flux attribution ──
if str(cfg["scaffold"]) == "txtl_model4_three_state":
    print("\n[7] M4 FLUX ATTRIBUTION  (protein ≈ ∫v_TL·M dt · k_mat/(k_mat+k_degp))")
    print("   idx   src   ∫vTL·M   matFrac   pm_pred  (check)   pm_true")
    for j in list(si[::-1][:4]) + list(si[:4]):
        Li = int(lengths[j])
        vTL = theta[j, :Li, 1]; M = pred[j, :Li, 0]
        k_mat = theta[j, :Li, 3]; k_deg = theta[j, :Li, 4]
        I = np.sum(vTL * M * dt_np[j, :Li])
        mat_frac = np.mean(k_mat / (k_mat + k_deg + 1e-12))
        print(f"   {rows[j]['idx']:<5d} {rows[j]['source']:>3s}  {I:8.1f}  {mat_frac:6.2f}    "
              f"{rows[j]['pm_pred']:7.0f} ({I*mat_frac:7.0f})  {rows[j]['pm_true']:7.0f}")

# ── plots ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for s, c, lab in [(0, "tab:blue", "old"), (1, "tab:orange", "new")]:
    m = src_test == s
    ax[0].scatter(pm_true[m], pm_err[m], s=18, c=c, alpha=.6, label=lab)
ax[0].axhline(0, color="k", lw=.7); ax[0].set_xlabel("true pm-final"); ax[0].set_ylabel("signed err (pred-true)")
ax[0].set_title("pm error vs true magnitude"); ax[0].legend(); ax[0].grid(alpha=.25)
if np.isfinite(ood).any():
    ax[1].scatter(ood, np.abs(pm_err), s=18, c=src_test, cmap="coolwarm", alpha=.6)
    ax[1].set_xlabel("input-OOD distance"); ax[1].set_ylabel("|pm err|")
    ax[1].set_title("error vs OOD distance"); ax[1].grid(alpha=.25)
fig.tight_layout(); fig.savefig(diag_dir / "failure_overview.png", dpi=140); plt.close(fig)
print(f"\nwrote: {diag_dir/'endpoint_table.csv'}  and  {diag_dir/'failure_overview.png'}")
