"""Build a combined NPZ that keeps each sample on its NATIVE time grid.

Uses the per-sample dt support added to ODEDataset (train.py). No interpolation
of OLD → NEW or NEW → OLD on the real data — each sample's `dt_per_sample[i, :L_i]`
matches the original acquisition grid.

Sources:
    - OLD: datasets/cell-free/real_ivtt_exact_full_drop5000.npz
      → 691 real samples on 60s grid (K up to 1497, actual lengths up to ~1233)
    - NEW + synth: datasets/cell-free/txtl_combined.npz
      → 312 real (Nov 2025 wells) + 7488 synth on NEW ~160s grid (K=324)

Output variants (toggled by --include-* flags):
    --include-old                                  → OLD only (controls: per-sample dt sanity)
    --include-old --include-new                    → OLD + NEW real (no synth)
    --include-old --include-new --include-synth    → OLD + NEW + synth

Synth grid policy (--synth-on-grid):
    'old'    : resample synth NEW (160s) → OLD (60s) using sum-of-deltas for u_seq
               and linear interpolation for y_seq. User-requested default.
               Cost: more timesteps per synth sample (1497 vs 324) → 4.6× more
               compute per epoch on synth-included variants.
    'native' : keep synth on its native NEW 160s grid (no resampling).

Output NPZ fields (compatible with the existing ODEDataset loader):
    y0:              (N, P)
    u_seq:           (N, K_max, U)   — zero-padded past per-sample length
    y_seq:           (N, K_max, P)   — zero-padded past per-sample length
    t_obs:           (K_max + 1,)    — finest grid (60s) for backward references
    dt_per_sample:   (N, K_max)      — per-sample dt, zeros past lengths
    lengths:         (N,)            — per-sample valid step count
    z_expr:          (N,)            — 1=real, 0=synth
    source_label:    (N,)            — 'old' | 'new' | 'synth'
    control_indices, obs_indices, control_names, obs_names, n_states_full,
    n_params_full, theta_true                       — same as txtl_combined.npz

Usage:
    # Variant 1: OLD only
    python scripts/build_native_grids_npz.py --include-old \\
        --output datasets/cell-free/txtl_native_old_only.npz

    # Variant 2: OLD + NEW real
    python scripts/build_native_grids_npz.py --include-old --include-new \\
        --output datasets/cell-free/txtl_native_real_only.npz

    # Variant 3: OLD + NEW + synth (synth re-binned to 60s grid)
    python scripts/build_native_grids_npz.py --include-old --include-new --include-synth \\
        --output datasets/cell-free/txtl_native_combined.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


OLD_NPZ = "datasets/cell-free/real_ivtt_exact_full_drop5000.npz"
COMBINED_NPZ = "datasets/cell-free/txtl_combined.npz"

# OLD's native grid: 60s spacing, K up to 1497 (t_obs covers 0..89820s).
OLD_DT_SECONDS = 60.0
OLD_K_MAX = 1497


def _load_npz(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _pad_to_K(arr: np.ndarray, K_target: int, fill: float = 0.0) -> np.ndarray:
    """Pad the time axis (axis=1) of a (N, K, ...) array up to K_target with `fill`."""
    if arr.shape[1] >= K_target:
        return arr[:, :K_target]
    pad_shape = list(arr.shape)
    pad_shape[1] = K_target - arr.shape[1]
    pad = np.full(pad_shape, fill, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)


def _per_sample_dt_constant(N: int, dt_value: float, K_max: int, lengths: np.ndarray) -> np.ndarray:
    """Build (N, K_max) dt array where each sample has constant dt up to its length,
    then zeros past that. Used for OLD (constant 60s) and synth-on-OLD-grid."""
    out = np.zeros((N, K_max), dtype=np.float32)
    for i in range(N):
        L = int(lengths[i])
        out[i, :L] = dt_value
    return out


def _per_sample_dt_from_tobs(N: int, t_obs: np.ndarray, K_max: int, lengths: np.ndarray) -> np.ndarray:
    """Build (N, K_max) dt array where each sample uses np.diff(t_obs) up to its length."""
    base_dt = np.diff(t_obs).astype(np.float32)   # (K_native,)
    out = np.zeros((N, K_max), dtype=np.float32)
    for i in range(N):
        L = int(lengths[i])
        out[i, :L] = base_dt[:L]
    return out


def _resample_synth_to_old_grid(
    u_seq_new: np.ndarray, y_seq_new: np.ndarray,
    t_obs_new: np.ndarray, lengths_new: np.ndarray,
    t_obs_old: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample synth from the NEW ~160s irregular grid to the OLD 60s grid.

    u_seq: sum-of-deltas within each OLD bin [t_old[k], t_old[k+1]).
           Preserves total reagent additions.
    y_seq: linear interpolation onto t_old timestamps.

    Returns (u_seq_old, y_seq_old, lengths_old) all shaped to t_obs_old.
    For each sample, the new length is set to len(t_obs_old) - 1 (full grid)
    if the original length spanned the full t_obs_new, else proportionally clipped.
    """
    N, _, U = u_seq_new.shape
    _, _, P = y_seq_new.shape
    K_old = len(t_obs_old) - 1
    u_seq_old = np.zeros((N, K_old, U), dtype=np.float32)
    y_seq_old = np.zeros((N, K_old, P), dtype=np.float32)
    lengths_old = np.zeros(N, dtype=np.int64)

    # OLD bin edges (t_obs_old) and NEW timestamps (centered between t_obs_new[k]
    # and t_obs_new[k+1] — the per-step delta is recorded at each midpoint).
    new_t_mid = 0.5 * (t_obs_new[:-1] + t_obs_new[1:])  # (K_new,)

    for i in range(N):
        L_new = int(lengths_new[i])
        if L_new <= 0:
            continue
        t_max_sample = float(t_obs_new[L_new])

        # ---- u_seq: bin NEW deltas into OLD bins via t_new_mid membership.
        for k_new in range(L_new):
            t_mid = new_t_mid[k_new]
            # find OLD bin idx (vectorize later if too slow)
            k_old = int(np.searchsorted(t_obs_old[1:], t_mid, side="right"))
            if k_old < K_old:
                u_seq_old[i, k_old, :] += u_seq_new[i, k_new, :]

        # ---- y_seq: linear interp from (t_obs_new, y_seq_new) onto t_obs_old[1:].
        # We use t_obs_new[1:] as the timestamp of y_seq_new[k] (post-step value).
        t_y_new = t_obs_new[1:L_new + 1]
        for p_idx in range(P):
            y_seq_old[i, :, p_idx] = np.interp(
                t_obs_old[1:], t_y_new, y_seq_new[i, :L_new, p_idx],
                left=y_seq_new[i, 0, p_idx], right=y_seq_new[i, L_new - 1, p_idx],
            )

        # ---- length: largest k_old < K_old where t_obs_old[k_old + 1] <= t_max_sample
        L_old = int(np.searchsorted(t_obs_old, t_max_sample, side="right") - 1)
        lengths_old[i] = max(1, min(L_old, K_old))

    return u_seq_old, y_seq_old, lengths_old


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-old", action="store_true",
                    help="Include OLD legacy real samples (real_ivtt_exact_full_drop5000.npz).")
    ap.add_argument("--include-new", action="store_true",
                    help="Include NEW real samples from txtl_combined.npz (z_expr==1).")
    ap.add_argument("--include-synth", action="store_true",
                    help="Include synthetic no-go samples from txtl_combined.npz (z_expr==0).")
    ap.add_argument("--synth-on-grid", choices=["old", "native"], default="old",
                    help="'old' (default): resample synth to OLD 60s grid. "
                         "'native': keep synth on NEW ~160s grid.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if not (args.include_old or args.include_new or args.include_synth):
        sys.exit("nothing to include — pass at least one --include-* flag.")

    old_d = _load_npz(OLD_NPZ) if args.include_old else None
    combined_d = _load_npz(COMBINED_NPZ) if (args.include_new or args.include_synth) else None

    # Sanity-check schema compatibility between OLD and COMBINED.
    if old_d is not None and combined_d is not None:
        old_names = list(old_d["control_names"])
        new_names = list(combined_d["control_names"])
        if old_names != new_names:
            sys.exit(f"control_names mismatch between OLD and COMBINED npzs:\n  OLD: {old_names}\n  NEW: {new_names}")

    # OLD's t_obs grid is our reference because it's the finest (60s) and longest.
    t_obs_old = (old_d["t_obs"] if old_d is not None
                 else np.arange(OLD_K_MAX + 1, dtype=np.float32) * OLD_DT_SECONDS)
    K_max = len(t_obs_old) - 1

    pieces: list[dict] = []   # each: {'y0','u','y','dt_per','lengths','z_expr','source'}

    # ---- OLD on native 60s grid ----
    if args.include_old:
        N_old = old_d["y0"].shape[0]
        u_old = _pad_to_K(old_d["u_seq"].astype(np.float32), K_max)
        y_old = _pad_to_K(old_d["y_seq"].astype(np.float32), K_max)
        lengths_old_native = (old_d["lengths"].astype(np.int64) if "lengths" in old_d
                              else np.full(N_old, K_max, dtype=np.int64))
        dt_old = _per_sample_dt_constant(N_old, OLD_DT_SECONDS, K_max, lengths_old_native)
        z_old = np.ones(N_old, dtype=np.int64)
        src_old = np.array(["old"] * N_old, dtype=object)
        pieces.append({
            "y0": old_d["y0"].astype(np.float32),
            "u":  u_old, "y": y_old,
            "dt_per": dt_old, "lengths": lengths_old_native,
            "z_expr": z_old, "source": src_old,
        })
        print(f"OLD legacy:  N={N_old}, lengths∈[{lengths_old_native.min()},{lengths_old_native.max()}], dt=60s")

    # ---- NEW real on native ~160s grid ----
    # NEW is left raw (un-blanked) on purpose: OLD's per-sample, per-channel
    # min-subtraction is reproduced at runtime via subtract_channel_min in the
    # trainer, so both regimes are gated identically. Don't bake baseline
    # subtraction into the NPZ.
    if args.include_new:
        z_combined = combined_d["z_expr"].astype(np.int64)
        new_mask = z_combined == 1
        N_new = int(new_mask.sum())
        u_new_native = combined_d["u_seq"][new_mask].astype(np.float32)  # (N_new, K_new, U)
        y_new_native = combined_d["y_seq"][new_mask].astype(np.float32)
        lengths_new = (combined_d["lengths"][new_mask].astype(np.int64)
                       if "lengths" in combined_d
                       else np.full(N_new, u_new_native.shape[1], dtype=np.int64))
        t_obs_new = combined_d["t_obs"].astype(np.float32)
        # Pad time axis up to K_max (rest stays zero — masked out by lengths).
        u_new = _pad_to_K(u_new_native, K_max)
        y_new = _pad_to_K(y_new_native, K_max)
        dt_new = _per_sample_dt_from_tobs(N_new, t_obs_new, K_max, lengths_new)
        z_new = np.ones(N_new, dtype=np.int64)
        src_new = np.array(["new"] * N_new, dtype=object)
        pieces.append({
            "y0": combined_d["y0"][new_mask].astype(np.float32),
            "u":  u_new, "y": y_new,
            "dt_per": dt_new, "lengths": lengths_new,
            "z_expr": z_new, "source": src_new,
        })
        print(f"NEW real:    N={N_new}, lengths∈[{lengths_new.min()},{lengths_new.max()}], "
              f"dt≈{float(np.median(np.diff(t_obs_new))):.0f}s (irregular)")

    # ---- synth: optionally resample to OLD 60s grid, else keep native ----
    if args.include_synth:
        z_combined = combined_d["z_expr"].astype(np.int64)
        synth_mask = z_combined == 0
        N_synth = int(synth_mask.sum())
        u_synth_native = combined_d["u_seq"][synth_mask].astype(np.float32)
        y_synth_native = combined_d["y_seq"][synth_mask].astype(np.float32)
        lengths_synth_native = (combined_d["lengths"][synth_mask].astype(np.int64)
                                if "lengths" in combined_d
                                else np.full(N_synth, u_synth_native.shape[1], dtype=np.int64))
        t_obs_new = combined_d["t_obs"].astype(np.float32)

        if args.synth_on_grid == "old":
            print(f"Synth:       N={N_synth}, resampling 160s→60s grid (this can take a minute)...")
            u_synth_old, y_synth_old, lengths_synth_old = _resample_synth_to_old_grid(
                u_synth_native, y_synth_native, t_obs_new, lengths_synth_native, t_obs_old,
            )
            u_synth = _pad_to_K(u_synth_old, K_max)
            y_synth = _pad_to_K(y_synth_old, K_max)
            dt_synth = _per_sample_dt_constant(N_synth, OLD_DT_SECONDS, K_max, lengths_synth_old)
            lengths_synth_final = lengths_synth_old
            print(f"              after resample: lengths∈[{lengths_synth_old.min()},{lengths_synth_old.max()}], dt=60s")
        else:
            # Keep on native NEW grid.
            u_synth = _pad_to_K(u_synth_native, K_max)
            y_synth = _pad_to_K(y_synth_native, K_max)
            dt_synth = _per_sample_dt_from_tobs(N_synth, t_obs_new, K_max, lengths_synth_native)
            lengths_synth_final = lengths_synth_native
            print(f"Synth:       N={N_synth}, kept on NEW native grid (~160s)")

        z_synth = np.zeros(N_synth, dtype=np.int64)
        src_synth = np.array(["synth"] * N_synth, dtype=object)
        pieces.append({
            "y0": combined_d["y0"][synth_mask].astype(np.float32),
            "u":  u_synth, "y": y_synth,
            "dt_per": dt_synth, "lengths": lengths_synth_final,
            "z_expr": z_synth, "source": src_synth,
        })

    # ---- u_open: tube-opening signal at t≈22h for NEW samples ─────────────
    # Hard-coded: in the NEW batch (2025-11), tubes were opened ~22h into
    # each experiment, allowing O2 to re-enter and dark protein to mature.
    # Encoded as an extra (13th) column in u_seq with control_name "u_open",
    # so the existing make_u_to_y_jump machinery wires it to whichever scaffold
    # state declares "u_open" in its control_state_map (Model 9: tube_opened).
    # OLD and synth pieces get all-zero u_open (no tube-opening event).
    OPEN_TIME_SECONDS = 22.0 * 3600.0
    for p in pieces:
        N_p, K_p, _ = p["u"].shape
        u_open_col = np.zeros((N_p, K_p, 1), dtype=np.float32)
        if p["source"][0] == "new":
            for i in range(N_p):
                L = int(p["lengths"][i])
                if L <= 0:
                    continue
                t_cum = np.cumsum(p["dt_per"][i, :L])
                k_open = int(np.argmin(np.abs(t_cum - OPEN_TIME_SECONDS)))
                u_open_col[i, k_open, 0] = 1.0
        # Append the new column to u_seq along the feature axis.
        p["u"] = np.concatenate([p["u"], u_open_col], axis=-1)

    # ---- concat all pieces ----
    out = {
        "y0":           np.concatenate([p["y0"] for p in pieces], axis=0),
        "u_seq":        np.concatenate([p["u"] for p in pieces], axis=0),
        "y_seq":        np.concatenate([p["y"] for p in pieces], axis=0),
        "dt_per_sample": np.concatenate([p["dt_per"] for p in pieces], axis=0),
        "lengths":      np.concatenate([p["lengths"] for p in pieces], axis=0),
        "z_expr":       np.concatenate([p["z_expr"] for p in pieces], axis=0),
        "source_label": np.concatenate([p["source"] for p in pieces], axis=0),
        "t_obs":        t_obs_old.astype(np.float32),
    }

    # Schema metadata copied from a source npz (prefer COMBINED so the file is
    # interchangeable with txtl_combined_plus_legacy.npz).
    schema_src = combined_d if combined_d is not None else old_d
    for k in ("control_indices", "obs_indices", "control_names", "obs_names",
              "names_full", "n_states_full", "n_params_full", "theta_true"):
        if k in schema_src:
            out[k] = schema_src[k]

    # Append "u_open" to control_names (the 13th feature column we just added).
    cn = np.array(list(out["control_names"]) + ["u_open"], dtype=out["control_names"].dtype)
    out["control_names"] = cn
    # Extend control_indices by one (placeholder index; scaffolds key by name).
    out["control_indices"] = np.array(list(out["control_indices"]) + [int(out["control_indices"].max()) + 1], dtype=out["control_indices"].dtype)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print()
    print(f"Wrote {out_path}")
    print(f"  N total = {out['y0'].shape[0]}  ({(out['z_expr']==1).sum()} real + {(out['z_expr']==0).sum()} synth)")
    print(f"  K_max   = {K_max}")
    print(f"  lengths∈[{out['lengths'].min()}, {out['lengths'].max()}]")
    print(f"  sources: " + ", ".join(f"{s}={int((out['source_label']==s).sum())}"
                                     for s in np.unique(out['source_label'])))


if __name__ == "__main__":
    main()
