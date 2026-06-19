"""Plot 4 OLD + 4 NEW mm/pm trajectories side-by-side, post-gate.

Quick visual sanity check that OLD and NEW look like the same kind of reaction
after per-sample min-subtraction is applied (mirrors the runtime gate).
"""
import numpy as np
import matplotlib.pyplot as plt

DS = "datasets/cell-free/txtl_native_real_only.npz"
MM_IDX = 3
PM_IDX = 5
N_PER = 4

d = np.load(DS, allow_pickle=True)
y_seq = d["y_seq"].astype(np.float32)
dt = d["dt_per_sample"].astype(np.float32)
lengths = d["lengths"].astype(np.int64)
src = np.array([str(s) for s in d["source_label"]])

rng = np.random.default_rng(7)
old_pool = np.where(src == "old")[0]
new_pool = np.where(src == "new")[0]
old_idx = rng.choice(old_pool, N_PER, replace=False)
new_idx = rng.choice(new_pool, N_PER, replace=False)

def gate(y, L):
    out = y[:L].copy()
    out -= out.min(axis=0, keepdims=True)
    return out

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)

for i, ix in enumerate(old_idx):
    L = int(lengths[ix])
    yg = gate(y_seq[ix], L)
    t = np.cumsum(np.concatenate([[0.0], dt[ix, :L - 1]])) / 3600.0
    axes[0, 0].plot(t, yg[:, MM_IDX], label=f"old {ix}", alpha=0.8)
    axes[1, 0].plot(t, yg[:, PM_IDX], label=f"old {ix}", alpha=0.8)

for i, ix in enumerate(new_idx):
    L = int(lengths[ix])
    yg = gate(y_seq[ix], L)
    t = np.cumsum(np.concatenate([[0.0], dt[ix, :L - 1]])) / 3600.0
    axes[0, 1].plot(t, yg[:, MM_IDX], label=f"new {ix}", alpha=0.8)
    axes[1, 1].plot(t, yg[:, PM_IDX], label=f"new {ix}", alpha=0.8)

axes[0, 0].set_title("OLD — mm (post-gate)")
axes[0, 1].set_title("NEW — mm (post-gate)")
axes[1, 0].set_title("OLD — pm (post-gate)")
axes[1, 1].set_title("NEW — pm (post-gate)")
for ax in axes.flat:
    ax.set_xlabel("time (h)")
    ax.set_ylabel("RFU (blanked)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)

# Also: overlay all 8 on one panel each, normalized by their own max, to see SHAPE.
fig.tight_layout()
out_path = "results/old_vs_new_traj.png"
fig.savefig(out_path, dpi=140)
print(f"wrote {out_path}")

# Second figure: shape comparison (normalized to peak)
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
for ix in old_idx:
    L = int(lengths[ix]); yg = gate(y_seq[ix], L)
    t = np.cumsum(np.concatenate([[0.0], dt[ix, :L - 1]])) / 3600.0
    pm = yg[:, PM_IDX]; mm = yg[:, MM_IDX]
    if pm.max() > 1: axes2[1].plot(t, pm / pm.max(), 'b-', alpha=0.5, label='old' if ix == old_idx[0] else None)
    if mm.max() > 1: axes2[0].plot(t, mm / mm.max(), 'b-', alpha=0.5, label='old' if ix == old_idx[0] else None)
for ix in new_idx:
    L = int(lengths[ix]); yg = gate(y_seq[ix], L)
    t = np.cumsum(np.concatenate([[0.0], dt[ix, :L - 1]])) / 3600.0
    pm = yg[:, PM_IDX]; mm = yg[:, MM_IDX]
    if pm.max() > 1: axes2[1].plot(t, pm / pm.max(), 'r-', alpha=0.5, label='new' if ix == new_idx[0] else None)
    if mm.max() > 1: axes2[0].plot(t, mm / mm.max(), 'r-', alpha=0.5, label='new' if ix == new_idx[0] else None)
axes2[0].set_title("mm shape (normalized to peak)")
axes2[1].set_title("pm shape (normalized to peak)")
for ax in axes2: ax.set_xlabel("time (h)"); ax.grid(alpha=0.3); ax.legend()
fig2.tight_layout()
fig2.savefig("results/old_vs_new_shape.png", dpi=140)
print("wrote results/old_vs_new_shape.png")
