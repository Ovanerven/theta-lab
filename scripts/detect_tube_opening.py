"""Detect the tube-opening knee in each NEW pm trace via slope-acceleration.

Knee = the timepoint k where slope(next window) / slope(prev window)
is maximized — i.e. where the trace accelerates the most. This matches
the eye's perception of "kink" better than peak-slope or fixed-threshold.

Algorithm (per sample):
  1. Smooth pm with a moving average (window=9).
  2. For each candidate k with t[k] in [t_min, t_max]:
     - left_slope  = mean dpm/dt over (t[k]-W_HOURS, t[k]]
     - right_slope = mean dpm/dt over [t[k], t[k]+W_HOURS)
     - score = right_slope - left_slope        (additive acceleration)
  3. Knee = argmax score.
"""
import numpy as np
import matplotlib.pyplot as plt

DS = "datasets/cell-free/txtl_native_real_only.npz"
PM_IDX = 5
SMOOTH_WIN = 9
W_HOURS = 5.0              # left/right window for slope comparison
T_MIN_HOURS = 8.0          # earliest plausible knee
T_MAX_HOURS = 40.0         # latest plausible knee

d = np.load(DS, allow_pickle=True)
src = np.array([str(s) for s in d["source_label"]])
y_seq = d["y_seq"].astype(np.float32)
dt = d["dt_per_sample"].astype(np.float32)
lengths = d["lengths"].astype(np.int64)

def find_knee(pm: np.ndarray, dt_sample: np.ndarray):
    L = len(pm)
    if L < 2 * SMOOTH_WIN:
        return None
    kernel = np.ones(SMOOTH_WIN) / SMOOTH_WIN
    pm_s = np.convolve(pm, kernel, mode="same")
    slope = np.diff(pm_s) / np.maximum(dt_sample[:L - 1], 1e-6)
    t = np.cumsum(np.concatenate([[0.0], dt_sample[:L - 1]])) / 3600.0   # hours, len=L
    # For each candidate k, compute left/right window means of slope
    W_SEC = W_HOURS * 3600.0
    best_k = -1; best_score = -np.inf
    best_left = best_right = 0.0
    for k in range(1, L - 1):
        if t[k] < T_MIN_HOURS or t[k] > T_MAX_HOURS:
            continue
        # left window: indices where t in (t[k]-W, t[k]]
        left_mask = (t[:-1] > t[k] - W_HOURS) & (t[:-1] <= t[k])
        right_mask = (t[:-1] > t[k]) & (t[:-1] <= t[k] + W_HOURS)
        if left_mask.sum() < 2 or right_mask.sum() < 2:
            continue
        left_slope = float(slope[left_mask].mean())
        right_slope = float(slope[right_mask].mean())
        # additive acceleration (units pm/sec). Penalize when both sides are tiny.
        score = right_slope - left_slope
        if score > best_score:
            best_score = score
            best_k = k
            best_left = left_slope; best_right = right_slope
    if best_k < 0:
        return None
    return {
        "k": best_k,
        "t_h": float(t[best_k]),
        "left_slope": best_left,
        "right_slope": best_right,
        "accel": best_score,
        "pm_at_knee": float(pm_s[best_k]),
        "pm_max": float(pm.max()),
    }

new_idx = np.where(src == "new")[0]
knees = []
for ix in new_idx:
    L = int(lengths[ix])
    pm = y_seq[ix, :L, PM_IDX].copy()
    pm -= pm.min()
    res = find_knee(pm, dt[ix])
    if res is not None:
        res["idx"] = int(ix); res["L"] = L
        knees.append(res)

print(f"Detected knee in {len(knees)} / {len(new_idx)} NEW samples")
times = np.array([k["t_h"] for k in knees])
print("\nKnee-time distribution (hours):")
for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
    print(f"  q{int(q*100):>2}: {np.quantile(times, q):.2f}h")
print(f"  mean: {times.mean():.2f}h   std: {times.std():.2f}h")

pm_maxes = np.array([k["pm_max"] for k in knees])
print("\nKnee-time conditional on pm_max (filter out tiny-signal samples):")
for thresh in [50, 100, 200, 500]:
    m = pm_maxes > thresh
    if m.sum() > 0:
        t_sub = times[m]
        print(f"  pm_max > {thresh:4d}: N={m.sum():3d}  median knee = {np.median(t_sub):.2f}h  "
              f"q25-q75=[{np.quantile(t_sub, 0.25):.2f}, {np.quantile(t_sub, 0.75):.2f}]h")

# Look at specific samples the user inspected: 961, 939, 717, 872, 938, 746, 747, 763
print("\nDirect check on user-inspected samples:")
by_idx = {k["idx"]: k for k in knees}
for ix in [763, 961, 939, 747, 746, 717, 872, 938]:
    k = by_idx.get(ix)
    if k:
        print(f"  sample {ix}: knee @ {k['t_h']:.1f}h  pm_max={k['pm_max']:.0f}  "
              f"left={k['left_slope']*3600:.2f} pm/h  right={k['right_slope']*3600:.2f} pm/h")
    else:
        print(f"  sample {ix}: no knee found")

# Plot the same 8 samples the user looked at, with new detection
fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
for i, ix in enumerate([763, 961, 939, 747, 746, 717, 872, 938]):
    L = int(lengths[ix])
    pm = y_seq[ix, :L, PM_IDX].copy(); pm -= pm.min()
    t = np.cumsum(np.concatenate([[0.0], dt[ix, :L - 1]])) / 3600.0
    ax = axes[i // 4, i % 4]
    ax.plot(t, pm, "b-")
    k = by_idx.get(ix)
    if k:
        ax.axvline(k["t_h"], color="r", linestyle="--", alpha=0.7,
                   label=f"knee @ {k['t_h']:.1f}h")
    ax.set_title(f"NEW sample {ix}  L={L}")
    ax.set_xlabel("time (h)"); ax.set_ylabel("pm"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
fig.tight_layout()
fig.savefig("results/tube_opening_knees.png", dpi=140)
print("\nwrote results/tube_opening_knees.png")

# Histogram of knee times for samples with real signal
fig2, ax = plt.subplots(figsize=(8, 4))
ax.hist(times[pm_maxes > 100], bins=40, alpha=0.7, color="steelblue", label="pm_max > 100")
ax.hist(times[pm_maxes > 500], bins=40, alpha=0.7, color="darkorange", label="pm_max > 500")
ax.set_xlabel("detected knee time (h)"); ax.set_ylabel("count")
ax.set_title("Detected tube-opening times across NEW samples (acceleration-based)")
ax.legend(); ax.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("results/tube_opening_hist.png", dpi=140)
print("wrote results/tube_opening_hist.png")
