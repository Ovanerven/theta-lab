"""Adaptive event-preserving downsampling of OLD trajectories to ~match NEW's grid.

OLD is uniform 60s (~1318 steps); NEW is fine-early/coarse-600s-late (~270 steps).
We coarsen OLD the SAME way: keep 60s through an early fine window (boluses + early
mRNA dynamics), then decimate the smooth tail to 600s. This is PURE DECIMATION — we
only drop real measured points (every 10th in the tail), never fabricate (unlike the
upsampling that failed). Boluses all fire <5520s, inside the fine window, so their
timing/spacing is fully preserved.

Reports, per fine-window cutoff:
  - resulting OLD step count (vs new ~270)
  - VARIANCE PRESERVED: decimate -> linear-reconstruct to the full 60s grid -> R²
    vs the original trajectory, per channel (mm=3, pm=5). High R² = safe drop.
Run with WRITE=1 (env) to also write the chosen-cutoff dataset.
"""
import os, numpy as np

SRC = os.environ.get("SRC", "datasets/cell-free/txtl_native_real_only.npz")
OUT = os.environ.get("OUT", "datasets/cell-free/txtl_native_real_only_coarsenold.npz")
TAIL_STRIDE = 10          # 10 * 60s = 600s, matches new's tail grid
MM, PM = 3, 5             # dataset obs channels

d = np.load(SRC, allow_pickle=True)
src = np.array([str(s) for s in d["source_label"]])
L = d["lengths"]; dt = d["dt_per_sample"]; y = d["y_seq"]; u = d["u_seq"]
oldi = np.where(src == "old")[0]            # ONLY old (60s) gets coarsened
otheri = np.where(src != "old")[0]          # new + synth already on 600s grid → unchanged
newi = otheri                                # alias for the report block below


def keep_indices(i, t_fine):
    """Indices to keep for sample i: all steps with time<=t_fine, then every
    TAIL_STRIDE-th step after. Returns sorted unique index array."""
    Li = int(L[i]); t = np.cumsum(dt[i, :Li])
    fine = np.where(t <= t_fine)[0]
    n_fine = len(fine)
    tail = np.arange(n_fine, Li)[::TAIL_STRIDE]   # every 10th in the tail
    # ALWAYS keep the final timestep so pm-final (endpoint metric) is the exact
    # same target as the original — decimation must not move the last point.
    return np.unique(np.concatenate([fine, tail, [Li - 1]])).astype(int)


def var_preserved(i, idx, ch):
    """R² of linear reconstruction (from kept pts) vs full trajectory, channel ch."""
    Li = int(L[i]); t = np.cumsum(dt[i, :Li]); full = y[i, :Li, ch]
    recon = np.interp(t, t[idx], full[idx])
    ss_tot = np.sum((full - full.mean()) ** 2)
    if ss_tot < 1e-9:
        return 1.0
    return 1.0 - np.sum((full - recon) ** 2) / ss_tot


print(f"OLD n={len(oldi)} (median {int(np.median(L[oldi]))} steps @60s)  |  "
      f"NEW n={len(newi)} (median {int(np.median(L[newi]))} steps)\n")
print(f"{'t_fine':>7} | {'old steps':>9} | {'var preserved mm':>16} | {'var preserved pm':>16}")
for t_fine in (6000, 10000, 15000, 20000):
    steps = []; vmm = []; vpm = []
    for i in oldi:
        idx = keep_indices(i, t_fine)
        steps.append(len(idx))
        vmm.append(var_preserved(i, idx, MM))
        vpm.append(var_preserved(i, idx, PM))
    print(f"{t_fine:>7} | {int(np.median(steps)):>9} | "
          f"med {np.median(vmm):.4f} p10 {np.percentile(vmm,10):.4f} | "
          f"med {np.median(vpm):.4f} p10 {np.percentile(vpm,10):.4f}")

# ── optionally write the dataset for the chosen cutoff ──
if os.environ.get("WRITE"):
    T_FINE = int(os.environ.get("T_FINE", "10000"))
    print(f"\n[WRITE] building {OUT} with t_fine={T_FINE} ...")
    N = len(src)
    new_lengths = L.copy()
    kept_per = {}
    for i in oldi:
        kept_per[i] = keep_indices(i, T_FINE)
        new_lengths[i] = len(kept_per[i])
    for i in newi:
        kept_per[i] = np.arange(int(L[i]))   # new unchanged
    Kmax = int(new_lengths.max())
    u2 = np.zeros((N, Kmax, u.shape[2]), u.dtype)
    y2 = np.zeros((N, Kmax, y.shape[2]), y.dtype)
    dt2 = np.zeros((N, Kmax), dt.dtype)
    for i in range(N):
        idx = kept_per[i]; m = len(idx); Li = int(L[i])
        t = np.cumsum(dt[i, :Li])
        kt = t[idx]
        ndt = np.empty(m, dt.dtype); ndt[0] = kt[0]; ndt[1:] = np.diff(kt)
        u2[i, :m] = u[i, idx]
        y2[i, :m] = y[i, idx]
        dt2[i, :m] = ndt
    out = {k: d[k] for k in d.files}
    out["u_seq"] = u2; out["y_seq"] = y2; out["dt_per_sample"] = dt2
    out["lengths"] = new_lengths
    out["t_obs"] = np.cumsum(dt2[int(np.argmax(new_lengths))][:Kmax]).astype(np.float32)
    np.savez(OUT, **out)
    print(f"[WRITE] done. OLD median steps {int(np.median(L[oldi]))} -> {int(np.median(new_lengths[oldi]))}; "
          f"NEW median {int(np.median(new_lengths[newi]))}; Kmax {u.shape[1]} -> {Kmax}")
