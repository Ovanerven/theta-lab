"""Diagnose why B_native doesn't generalize: compare train vs val vs test
distributions for OLD-source and NEW-source samples separately.

Output:
  1. Split composition (how many OLD/NEW per split)
  2. Per-(split × source) endpoint distributions: pm_final, mm_max
  3. Per-(split × source) input distributions: y0 channels, total bolus per reagent
  4. Cross-distribution similarity checks (mean/median/std/quantiles)
  5. Outlier flags: samples > 3σ from their source-group median

If NEW-train and NEW-val cover different regions of pm/mm space, the model
can't learn to predict NEW val even if it memorizes NEW train. Likewise for
recipe (u0/cumulative bolus) shift.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

DS_PATH = Path("datasets/cell-free/txtl_native_real_only.npz")
EXP_DIR = Path("experiments/txtl_data_axis_native_gru/20260524_110742_B_native_real")

d = np.load(DS_PATH, allow_pickle=True)
y_seq = d["y_seq"].astype(np.float32)
u_seq = d["u_seq"].astype(np.float32)
y0 = d["y0"].astype(np.float32)
lengths = d["lengths"].astype(np.int64)
src = np.array([str(s) for s in d["source_label"]])
obs_names = list(d["obs_names"])
ctrl_names = list(d["control_names"])

# Apply runtime min-gate on cols [3, 5] to mirror training behaviour
MM_IDX, PM_IDX = 3, 5
def gated_endpoint(i: int):
    L = int(lengths[i])
    mm_trace = y_seq[i, :L, MM_IDX]
    pm_trace = y_seq[i, :L, PM_IDX]
    mm_g = mm_trace - mm_trace.min()
    pm_g = pm_trace - pm_trace.min()
    return float(pm_g[-1]), float(mm_g.max())

pm_finals = np.array([gated_endpoint(i)[0] for i in range(len(y0))])
mm_maxes  = np.array([gated_endpoint(i)[1] for i in range(len(y0))])

# Load the split that B_native used
split = np.load(EXP_DIR / "split.npz")
train_idx = split["train_idx"].astype(int)
val_idx   = split["val_idx"].astype(int)
test_idx  = split["test_idx"].astype(int)

def src_count(idx_arr):
    return (int((src[idx_arr] == "old").sum()), int((src[idx_arr] == "new").sum()))

print("=" * 90)
print("SPLIT COMPOSITION")
print("=" * 90)
print(f"{'split':>8s} {'N':>5s} {'OLD':>6s} {'NEW':>6s} {'%new':>6s}")
for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
    o, n = src_count(idx)
    print(f"{name:>8s} {len(idx):>5d} {o:>6d} {n:>6d} {100*n/(o+n):>6.1f}%")

print()
print("=" * 90)
print("ENDPOINT DISTRIBUTIONS (post-gate) per (split × source)")
print("=" * 90)

def stats(name, arr):
    if len(arr) < 2:
        return f"  {name:25s}: n={len(arr)} <too few>"
    q = np.quantile(arr, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return (f"  {name:25s}: n={len(arr):3d}  "
            f"min={q[0]:7.1f}  q05={q[1]:7.1f}  med={q[3]:7.1f}  q95={q[5]:7.1f}  max={q[6]:7.1f}  "
            f"mean={arr.mean():7.1f}±{arr.std():6.1f}")

for metric_name, metric in [("pm_final", pm_finals), ("mm_max", mm_maxes)]:
    print(f"\n{metric_name}:")
    for source in ["old", "new"]:
        for split_name, sidx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            mask = src[sidx] == source
            arr = metric[sidx][mask]
            print(stats(f"{source:>3s}-{split_name:>5s}", arr))

print()
print("=" * 90)
print("KOLMOGOROV-SMIRNOV: NEW-train vs NEW-val/test  (p<0.05 → distributions differ)")
print("=" * 90)
try:
    from scipy.stats import ks_2samp
    for metric_name, metric in [("pm_final", pm_finals), ("mm_max", mm_maxes)]:
        new_tr = metric[train_idx][src[train_idx] == "new"]
        new_va = metric[val_idx][src[val_idx] == "new"]
        new_te = metric[test_idx][src[test_idx] == "new"]
        ks_va, p_va = ks_2samp(new_tr, new_va)
        ks_te, p_te = ks_2samp(new_tr, new_te)
        print(f"  {metric_name}:")
        print(f"    NEW-train vs NEW-val:  KS={ks_va:.3f}  p={p_va:.4f}  {'DIFFERENT' if p_va < 0.05 else 'similar'}")
        print(f"    NEW-train vs NEW-test: KS={ks_te:.3f}  p={p_te:.4f}  {'DIFFERENT' if p_te < 0.05 else 'similar'}")
    print()
    print("  (And the OLD baseline for comparison:)")
    for metric_name, metric in [("pm_final", pm_finals), ("mm_max", mm_maxes)]:
        old_tr = metric[train_idx][src[train_idx] == "old"]
        old_va = metric[val_idx][src[val_idx] == "old"]
        ks, p = ks_2samp(old_tr, old_va)
        print(f"  {metric_name} OLD-train vs OLD-val: KS={ks:.3f}  p={p:.4f}  {'DIFFERENT' if p < 0.05 else 'similar'}")
except ImportError:
    print("  scipy not available — skipping KS test")

print()
print("=" * 90)
print("INPUT (cumulative bolus per reagent) — NEW-train vs NEW-val")
print("=" * 90)
# Cumulative bolus per sample per reagent
cum = np.array([
    np.abs(u_seq[i, :int(lengths[i]), :12]).sum(axis=0)   # (12,)
    for i in range(len(y0))
])  # (N, 12)

print(f"\n{'reagent':>15s} {'NEW-tr mean±std':>22s} {'NEW-va mean±std':>22s} {'ratio':>8s} {'KS p':>10s}")
new_tr_idx = train_idx[src[train_idx] == "new"]
new_va_idx = val_idx[src[val_idx] == "new"]
try:
    from scipy.stats import ks_2samp
    for c, nm in enumerate(ctrl_names[:12]):
        tr = cum[new_tr_idx, c]; va = cum[new_va_idx, c]
        if tr.std() < 1e-6 and va.std() < 1e-6:
            continue
        ratio = (va.mean() + 1e-9) / (tr.mean() + 1e-9)
        ks, p = ks_2samp(tr, va) if len(tr) > 1 and len(va) > 1 else (np.nan, np.nan)
        print(f"{nm:>15s} {f'{tr.mean():8.1f}±{tr.std():7.1f}':>22s} "
              f"{f'{va.mean():8.1f}±{va.std():7.1f}':>22s} {ratio:>8.2f} {p:>10.4f}")
except ImportError:
    pass

print()
print("=" * 90)
print("OUTLIERS — NEW samples with pm_final > 3σ from NEW median")
print("=" * 90)
new_mask = src == "new"
new_pm = pm_finals[new_mask]
new_med = np.median(new_pm); new_mad = np.median(np.abs(new_pm - new_med)) + 1e-9
z = (new_pm - new_med) / (1.4826 * new_mad)
new_global_idx = np.where(new_mask)[0]
outlier_local = np.where(np.abs(z) > 3)[0]
print(f"NEW pm_final: median={new_med:.1f}, MAD-derived σ≈{1.4826*new_mad:.1f}")
print(f"Outliers (|z|>3): {len(outlier_local)} of {new_mask.sum()} NEW samples")
for li in outlier_local[:15]:
    gi = new_global_idx[li]
    in_tr = "TR" if gi in train_idx else ("VA" if gi in val_idx else "TE")
    print(f"  global idx {gi}  [{in_tr}]  pm_final={new_pm[li]:.1f}  z={z[li]:+.2f}")

print("\nDone.")
