"""Diagnose OLD vs NEW samples in txtl_native_real_only.npz.

Checks the three hypotheses:
  1. mm/pm endpoint scale mismatch between OLD and NEW
  2. Time-horizon mismatch (different total sim time per sample)
  3. Bolus / u_seq input-distribution mismatch
"""
import numpy as np

DS = "datasets/cell-free/txtl_native_real_only.npz"
MM_IDX = 3
PM_IDX = 5

d = np.load(DS, allow_pickle=True)
y_seq = d["y_seq"]            # (N, T, 7)
u_seq = d["u_seq"]            # (N, T, 12)
y0 = d["y0"]                  # (N, 7)
dt = d["dt_per_sample"]       # (N, T)
lengths = d["lengths"]        # (N,)
src = np.array([str(s) for s in d["source_label"]])

print(f"N total = {len(lengths)}")
print(f"source_label unique values: {np.unique(src)}")

# Try to partition OLD vs NEW from source labels
src_lower = np.array([s.lower() for s in src])
old_mask = np.array(["old" in s or "legacy" in s for s in src_lower])
new_mask = ~old_mask

print(f"\nOLD count: {old_mask.sum()}   NEW count: {new_mask.sum()}")
if old_mask.sum() == 0 or new_mask.sum() == 0:
    print("WARNING: could not split by source_label string — adjust the masking.")

def stats(name, arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        print(f"  {name}: <empty>")
        return
    q = np.quantile(arr, [0.0, 0.05, 0.5, 0.95, 1.0])
    print(f"  {name}: min={q[0]:.4g}  p05={q[1]:.4g}  median={q[2]:.4g}  p95={q[3]:.4g}  max={q[4]:.4g}  mean={arr.mean():.4g}")

# ─── 1. Endpoint mm / pm scale ──────────────────────────────────────────────
def endpoints(mask):
    pm_finals, mm_maxes = [], []
    idxs = np.where(mask)[0]
    for i in idxs:
        L = int(lengths[i])
        pm_finals.append(float(y_seq[i, L - 1, PM_IDX]))
        mm_maxes.append(float(np.max(y_seq[i, :L, MM_IDX])))
    return np.array(pm_finals), np.array(mm_maxes)

pm_old, mm_old = endpoints(old_mask)
pm_new, mm_new = endpoints(new_mask)

print("\n=== 1. ENDPOINT SCALES ===")
print(" pm_final (last-timepoint pm)")
stats("  OLD pm_final", pm_old)
stats("  NEW pm_final", pm_new)
print(" mm_max (max of mm over time)")
stats("  OLD mm_max", mm_old)
stats("  NEW mm_max", mm_new)

ratio_pm = (np.median(pm_new) + 1e-9) / (np.median(pm_old) + 1e-9)
ratio_mm = (np.median(mm_new) + 1e-9) / (np.median(mm_old) + 1e-9)
print(f"\n median(NEW pm_final) / median(OLD pm_final) = {ratio_pm:.3f}")
print(f" median(NEW mm_max)   / median(OLD mm_max)   = {ratio_mm:.3f}")
print(" ► if either ratio is far from 1 (say <0.5 or >2), units/scale differ.")

# Also: full y_seq distribution per channel
print("\n full pm-channel value distribution (all timepoints up to length)")
def all_obs_vals(mask, ch):
    vals = []
    for i in np.where(mask)[0]:
        L = int(lengths[i])
        vals.append(y_seq[i, :L, ch])
    return np.concatenate(vals) if vals else np.array([])
stats(" OLD pm all-time", all_obs_vals(old_mask, PM_IDX))
stats(" NEW pm all-time", all_obs_vals(new_mask, PM_IDX))
stats(" OLD mm all-time", all_obs_vals(old_mask, MM_IDX))
stats(" NEW mm all-time", all_obs_vals(new_mask, MM_IDX))

# ─── 2. Time-horizon ────────────────────────────────────────────────────────
print("\n=== 2. TIME HORIZON ===")
def total_time(mask):
    out = []
    for i in np.where(mask)[0]:
        L = int(lengths[i])
        # dt_per_sample is per-step; cumulative time to last obs = sum of dt[:L-1]
        out.append(float(dt[i, : max(L - 1, 0)].sum()))
    return np.array(out)
def step_dt(mask):
    out = []
    for i in np.where(mask)[0]:
        L = int(lengths[i])
        out.append(dt[i, : max(L - 1, 0)])
    return np.concatenate(out) if out else np.array([])

t_old = total_time(old_mask)
t_new = total_time(new_mask)
stats("  OLD total_time (s)", t_old)
stats("  NEW total_time (s)", t_new)
stats("  OLD step dt (s)",    step_dt(old_mask))
stats("  NEW step dt (s)",    step_dt(new_mask))
print("  lengths:")
stats("  OLD lengths", lengths[old_mask].astype(float))
stats("  NEW lengths", lengths[new_mask].astype(float))

# ─── 3. Input/bolus distribution ────────────────────────────────────────────
print("\n=== 3. INPUT (y0 + u_seq) DISTRIBUTION ===")
# y0
print(" y0 per-feature mean (cols 0..n)")
for c in range(y0.shape[1]):
    m_old = y0[old_mask, c].mean()
    m_new = y0[new_mask, c].mean()
    s_old = y0[old_mask, c].std()
    s_new = y0[new_mask, c].std()
    flag = ""
    if abs(m_old) + abs(m_new) > 1e-9:
        rel = abs(m_new - m_old) / (abs(m_old) + abs(m_new) + 1e-9)
        if rel > 0.3:
            flag = "  <-- shifted"
    print(f"   y0[{c}]: OLD μ={m_old:+.3g} σ={s_old:.3g}   NEW μ={m_new:+.3g} σ={s_new:.3g}{flag}")

# u_seq at t=0 (bolus often lives here)
print("\n u_seq[:, 0, :] per-feature mean (t=0 bolus)")
u0 = u_seq[:, 0, :]
for c in range(u0.shape[1]):
    m_old = u0[old_mask, c].mean()
    m_new = u0[new_mask, c].mean()
    s_old = u0[old_mask, c].std()
    s_new = u0[new_mask, c].std()
    flag = ""
    if abs(m_old) + abs(m_new) > 1e-9:
        rel = abs(m_new - m_old) / (abs(m_old) + abs(m_new) + 1e-9)
        if rel > 0.3:
            flag = "  <-- shifted"
    print(f"   u0[{c}]: OLD μ={m_old:+.3g} σ={s_old:.3g}   NEW μ={m_new:+.3g} σ={s_new:.3g}{flag}")

# Are there bolus dims that are zero in one set and nonzero in the other?
print("\n nonzero-fraction at t=0 (fraction of samples where |u0[c]| > 1e-9)")
for c in range(u0.shape[1]):
    nz_old = (np.abs(u0[old_mask, c]) > 1e-9).mean()
    nz_new = (np.abs(u0[new_mask, c]) > 1e-9).mean()
    flag = ""
    if abs(nz_old - nz_new) > 0.3:
        flag = "  <-- coverage mismatch"
    print(f"   u0[{c}]: OLD={nz_old:.2f}  NEW={nz_new:.2f}{flag}")

print("\nDone.")
