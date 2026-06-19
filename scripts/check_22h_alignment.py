"""Overlay t=22h on the same 8 NEW samples to see if it lands on the visible kink."""
import numpy as np
import matplotlib.pyplot as plt

DS = "datasets/cell-free/txtl_native_real_only.npz"
PM_IDX = 5
FIXED_OPEN_H = 22.0

d = np.load(DS, allow_pickle=True)
src = np.array([str(s) for s in d["source_label"]])
y_seq = d["y_seq"].astype(np.float32)
dt = d["dt_per_sample"].astype(np.float32)
lengths = d["lengths"].astype(np.int64)
u_open = d["u_open"].astype(np.float32)   # from build_native_grids_npz.py

samples = [763, 961, 939, 747, 746, 717, 872, 938]
fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
for i, ix in enumerate(samples):
    L = int(lengths[ix])
    pm = y_seq[ix, :L, PM_IDX].copy()
    pm -= pm.min()
    t = np.cumsum(np.concatenate([[0.0], dt[ix, :L - 1]])) / 3600.0
    ax = axes[i // 4, i % 4]
    ax.plot(t, pm, "b-", alpha=0.85)
    # Encoded opening index from the NPZ
    k_open_arr = np.where(u_open[ix, :L] > 0)[0]
    if len(k_open_arr) > 0:
        k_open = int(k_open_arr[0])
        t_encoded = float(t[k_open])
        ax.axvline(t_encoded, color="green", linestyle="-", alpha=0.9, lw=2,
                   label=f"u_open=1 @ {t_encoded:.1f}h\n(k_open={k_open})")
    ax.axvline(FIXED_OPEN_H, color="r", linestyle="--", alpha=0.6,
               label=f"target t={FIXED_OPEN_H}h")
    ax.set_title(f"NEW sample {ix}  L={L}  total={t[-1]:.1f}h")
    ax.set_xlabel("time (h)"); ax.set_ylabel("pm")
    ax.legend(fontsize=7, loc="best"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/check_22h_alignment.png", dpi=140)
print("wrote results/check_22h_alignment.png")

# Also: report what u_open evaluates to for each of those 8 samples
print()
print(f"{'idx':>4} {'L':>4} {'total_h':>7} {'k_open':>7} {'t_encoded(h)':>14}")
for ix in samples:
    L = int(lengths[ix])
    t_total = float(np.cumsum(dt[ix, :L - 1])[-1]) / 3600.0
    k_arr = np.where(u_open[ix, :L] > 0)[0]
    if len(k_arr) > 0:
        k = int(k_arr[0])
        t_cum = np.cumsum(dt[ix, :L]) / 3600.0
        t_enc = float(t_cum[k])
        print(f"{ix:>4} {L:>4} {t_total:>7.1f} {k:>7} {t_enc:>14.2f}")
    else:
        print(f"{ix:>4} {L:>4} {t_total:>7.1f}  (no u_open marker)")
