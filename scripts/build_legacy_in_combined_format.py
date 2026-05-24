"""Convert legacy parquet IVTT data into the same .npz schema as txtl_combined.npz.

Use case
--------
The legacy real-data parquet pairs under `datasets/cell-free/data_parsed_pruned/`
(produced by Bob's `parse_IVTT_data.py`) come from earlier 2025 plates. They were
shipped as `datasets/cell-free/real_ivtt_exact_full.npz` on a 60s/1497-step grid
with baseline subtraction baked in. The November 2025 plates were processed into
`datasets/cell-free/txtl_combined.npz` (160s irregular grid, 324 steps, raw
baselines, plus synthetic no-go samples). Those two npz files use the same
state/control schema (P=7, D_IN=12) but are NOT directly stackable because:

    1. Different time grids (60s/1497 vs irregular ~160s/324)
    2. OLD has baseline-subtracted obs; NEW is raw
    3. OLD has 0 synth, NEW has 7488 synth

This script resamples the legacy parquets onto NEW's t_obs grid, keeps the raw
baselines (so `subtract_channel_min` in the trainer can handle both consistently),
and emits the result with `z_expr=1` (all real) in the combined-npz schema.

Resampling rules
----------------
- y_seq (Broccoli, mCherry/divisor): linear interpolation to t_target.
- u_seq (per-step deltas of each reagent): SUM the OLD per-step deltas that fall
  in each NEW bin [t_target[k], t_target[k+1]). Sums preserve total reagent added.
- Controls absent from the parquet schema (e.g. older plates missing 3PGA, AA,
  NTPs) are backfilled with zeros — consistent with what
  build_txtl_combined_npz.py already does for those plates.

Usage
-----
    # Build standalone legacy npz in combined schema
    python scripts/build_legacy_in_combined_format.py \\
        --data-dir datasets/cell-free/data_parsed_pruned \\
        --reference-npz datasets/cell-free/txtl_combined.npz \\
        --output datasets/cell-free/txtl_legacy_in_combined.npz

    # OR: merge directly into a new combined npz alongside the existing 7800 rows
    python scripts/build_legacy_in_combined_format.py \\
        --data-dir datasets/cell-free/data_parsed_pruned \\
        --reference-npz datasets/cell-free/txtl_combined.npz \\
        --merge \\
        --output datasets/cell-free/txtl_combined_plus_legacy.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------- Schema (must match build_txtl_combined_npz.py) ----------------

STATE_NAMES = ["R", "O", "m", "mm", "p", "pm", "DNA"]
P = 7
DNA_IDX = 6
MM_IDX = 3
PM_IDX = 5
X0_INIT = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

CONTROL_NAMES = [
    "3PGA", "AA", "DNA c", "FB", "K-Glut", "Lysate 2%PEG",
    "Maltose", "Mg-Glut", "NTPs", "PEG8000", "T7RNAP", "water",
]
D_IN = len(CONTROL_NAMES)
DNA_C_IDX = CONTROL_NAMES.index("DNA c")

# Parquet input column conventions (from scripts/convert_real_to_npz.py)
TIME_COL = "Time_seconds"
MRNA_RFU_COL = "Broccoli [RFU]"
PROTEIN_RFU_COL = "mCherry [RFU]"
PM_OUTLIER_THRESHOLD = 5000.0


# ---------------- Resampling helpers ----------------

def _bin_deltas_to_points(
    t_src: np.ndarray, deltas_src: np.ndarray, t_points: np.ndarray
) -> np.ndarray:
    """Sum per-step deltas onto bins anchored at K time points.

    `t_points` is a length-K array of time points (e.g. the combined dataset's
    t_obs). Each OLD per-step delta j is assigned to bin k where
        k = max{i : t_points[i] <= t_src[j]}
    (most recent target time at or before t_src[j]). This preserves
    sum(deltas) across the resampling.
    """
    K = len(t_points)
    if len(t_src) == 0 or deltas_src.size == 0:
        return np.zeros(K, dtype=np.float32)
    out = np.zeros(K, dtype=np.float32)
    bin_of = np.searchsorted(t_points, t_src, side="right") - 1
    bin_of = np.clip(bin_of, 0, K - 1)
    np.add.at(out, bin_of, deltas_src.astype(np.float32))
    return out


def _interp_obs_to_grid(
    t_src: np.ndarray, y_src: np.ndarray, t_target: np.ndarray
) -> np.ndarray:
    """Linearly interpolate observed values onto t_target. Extrapolation = clamp."""
    if len(t_src) == 0:
        return np.zeros(len(t_target), dtype=np.float32)
    return np.interp(t_target, t_src, y_src, left=y_src[0], right=y_src[-1]).astype(np.float32)


# ---------------- Manifest / parquet iteration ----------------

def _iter_parquet_pairs(data_dir: Path):
    """Yield (experiment_key, inputs_path, outputs_path) in manifest order."""
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json missing in {data_dir}")
    with open(manifest_path) as fh:
        entries = json.load(fh)
    for key, entry in entries.items():
        inp = data_dir / entry["inputs_path"]
        out = data_dir / entry["outputs_path"]
        if inp.exists() and out.exists():
            yield key, inp, out
        else:
            print(f"  WARNING: missing parquet for {key} — skipping")


# ---------------- Conversion ----------------

def _build_one_experiment(
    key: str,
    inp_path: Path,
    out_path: Path,
    t_points: np.ndarray,         # time-point grid (length K, matches y_seq.shape[1])
    K: int,                       # number of points (== len(t_points) == y_seq.shape[1])
    mcherry_divisor: float,
    outlier_mode: str,
) -> dict | None:
    """Return a record dict (y0, u, y_seq, etc.) in combined schema, or None if dropped."""
    df_inp = pd.read_parquet(inp_path)
    df_out = pd.read_parquet(out_path)

    df_inp.columns = [str(c).strip() for c in df_inp.columns]
    df_out.columns = [str(c).strip() for c in df_out.columns]

    if TIME_COL not in df_out.columns:
        print(f"  skip {key}: outputs missing '{TIME_COL}'")
        return None
    if MRNA_RFU_COL not in df_out.columns or PROTEIN_RFU_COL not in df_out.columns:
        print(f"  skip {key}: outputs missing Broccoli/mCherry")
        return None

    t_src_out = df_out[TIME_COL].to_numpy(dtype=np.float32)
    broccoli = df_out[MRNA_RFU_COL].to_numpy(dtype=np.float32)
    mcherry = df_out[PROTEIN_RFU_COL].to_numpy(dtype=np.float32) / float(mcherry_divisor)

    # Outlier drop (matches OLD pipeline pm > 5000)
    if outlier_mode == "max" and float(mcherry.max()) > PM_OUTLIER_THRESHOLD:
        return None
    if outlier_mode == "last" and float(mcherry[-1]) > PM_OUTLIER_THRESHOLD:
        return None

    # Inputs: per-step deltas. The OLD convert script took df_inp[control_cols][:-1]
    # — so u_seq[k] drives [t_src[k], t_src[k+1]). We use t_src_inp[k] = t_src_out[k]
    # (input rows align with output time stamps; last input row was dropped).
    n_inp = len(df_inp)
    t_src_inp = t_src_out[: min(n_inp, len(t_src_out))]
    if len(t_src_inp) == 0:
        return None

    u_row = np.zeros((K, D_IN), dtype=np.float32)
    for c_idx, cname in enumerate(CONTROL_NAMES):
        if cname not in df_inp.columns:
            continue  # missing reagent → keep zeros (matches build_txtl_combined_npz behavior)
        deltas_src = df_inp[cname].to_numpy(dtype=np.float32)[: len(t_src_inp)]
        u_row[:, c_idx] = _bin_deltas_to_points(t_src_inp, deltas_src, t_points)

    # Observed trajectories interpolated directly to the target time POINTS
    # (matches build_txtl_combined_npz.py convention: y_seq[k] = obs at t_obs[k]).
    mm_grid = _interp_obs_to_grid(t_src_out, broccoli, t_points)
    pm_grid = _interp_obs_to_grid(t_src_out, mcherry, t_points)

    y0 = X0_INIT.copy()
    y0[MM_IDX] = float(broccoli[0])
    y0[PM_IDX] = float(mcherry[0])

    y_seq = np.zeros((K, P), dtype=np.float32)
    y_seq[:, MM_IDX] = mm_grid
    y_seq[:, PM_IDX] = pm_grid

    return {
        "experiment_id": key,
        "y0": y0,
        "u": u_row,
        "y_seq": y_seq,
        "dna_total": float(df_inp.get("DNA c", pd.Series([0.0])).astype(float).sum()),
    }


def _build_combined_arrays(records: list[dict], t_points: np.ndarray) -> dict:
    """Stack per-experiment dicts into combined-npz arrays.

    `t_points` is the time-point grid (length K, same as y_seq.shape[1]).
    `lengths[i] = K` for every row (legacy data is all full-length after resampling).
    """
    N = len(records)
    K = int(records[0]["u"].shape[0])  # number of time points
    y0 = np.stack([r["y0"] for r in records], axis=0).astype(np.float32)
    u_seq = np.stack([r["u"] for r in records], axis=0).astype(np.float32)
    y_seq = np.stack([r["y_seq"] for r in records], axis=0).astype(np.float32)
    lengths = np.full(N, K, dtype=np.int64)
    z_expr = np.ones(N, dtype=np.int64)
    class_label = np.array(["real"] * N, dtype="<U32")
    experiment_id = np.array([r["experiment_id"] for r in records], dtype="<U128")
    dna_totals = np.array([r["dna_total"] for r in records], dtype=np.float32)

    obs_indices = np.arange(P, dtype=np.int64)
    control_indices = np.arange(P, P + D_IN, dtype=np.int64)
    control_indices[DNA_C_IDX] = DNA_IDX

    return dict(
        y0=y0, u_seq=u_seq, y_seq=y_seq,
        t_obs=t_points.astype(np.float32),
        control_indices=control_indices,
        obs_indices=obs_indices,
        names_full=np.array(STATE_NAMES + CONTROL_NAMES, dtype="<U32"),
        control_names=np.array(CONTROL_NAMES, dtype="<U32"),
        obs_names=np.array(STATE_NAMES, dtype="<U32"),
        n_states_full=np.int64(P),
        n_params_full=np.int64(0),
        theta_true=np.zeros(0, dtype=np.float32),
        lengths=lengths,
        reagent_scaling=np.array("none", dtype="<U16"),
        dna_mode=np.array("raw", dtype="<U16"),
        obs_align=np.array("current", dtype="<U16"),
        outlier_mode=np.array("max", dtype="<U16"),
        dna_totals=dna_totals,
        z_expr=z_expr,
        class_label=class_label,
        experiment_id=experiment_id,
    )


def _concat_with_existing(legacy: dict, ref_npz: Path) -> dict:
    """Concatenate the legacy arrays with txtl_combined.npz's content."""
    ref = np.load(str(ref_npz), allow_pickle=True)
    if not np.array_equal(ref["t_obs"].astype(np.float32), legacy["t_obs"].astype(np.float32)):
        raise ValueError("t_obs mismatch — legacy was resampled onto a different grid than reference.")
    if not np.array_equal(np.array(ref["control_names"]).astype(str),
                          np.array(legacy["control_names"]).astype(str)):
        raise ValueError("control_names mismatch between legacy and reference.")

    out = dict(legacy)  # start with legacy field set; we'll concat the stacked arrays
    for k in ["y0", "u_seq", "y_seq"]:
        out[k] = np.concatenate([ref[k], legacy[k]], axis=0).astype(np.float32)
    out["lengths"] = np.concatenate([ref["lengths"], legacy["lengths"]], axis=0).astype(np.int64)
    out["dna_totals"] = np.concatenate(
        [ref["dna_totals"], legacy["dna_totals"]], axis=0
    ).astype(np.float32)
    # z_expr / class_label / experiment_id — fall back to defaults if reference lacks them
    ref_z = ref["z_expr"] if "z_expr" in ref.files else np.ones(ref["y0"].shape[0], dtype=np.int64)
    out["z_expr"] = np.concatenate([ref_z.astype(np.int64), legacy["z_expr"]], axis=0)
    ref_cl = (ref["class_label"] if "class_label" in ref.files
              else np.array(["real"] * ref["y0"].shape[0], dtype="<U32"))
    out["class_label"] = np.concatenate([ref_cl.astype(str), legacy["class_label"]], axis=0)
    ref_eid = (ref["experiment_id"] if "experiment_id" in ref.files
               else np.array([f"REF_{i}" for i in range(ref["y0"].shape[0])], dtype="<U128"))
    out["experiment_id"] = np.concatenate(
        [ref_eid.astype(str), legacy["experiment_id"]], axis=0
    )
    # carry over scalars/metadata from reference (control_indices etc. must match)
    for k in ["control_indices", "obs_indices", "names_full", "control_names",
              "obs_names", "n_states_full", "n_params_full", "theta_true", "t_obs"]:
        if k in ref.files:
            out[k] = ref[k]
    return out


# ---------------- main ----------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/cell-free/data_parsed_pruned"))
    parser.add_argument("--reference-npz", type=Path, required=True,
                        help="Existing combined npz whose t_obs / control_names define the target schema.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mcherry-divisor", type=float, default=2.0)
    parser.add_argument("--outlier-mode", type=str, default="max", choices=["max", "last", "none"])
    parser.add_argument("--merge", action="store_true",
                        help="Concatenate legacy records with the reference npz before saving.")
    args = parser.parse_args()

    ref = np.load(str(args.reference_npz), allow_pickle=True)
    t_points = ref["t_obs"].astype(np.float32)         # length K (time POINTS)
    K = int(ref["y_seq"].shape[1])
    if len(t_points) != K:
        raise ValueError(
            f"Reference npz schema mismatch: len(t_obs)={len(t_points)} but "
            f"y_seq.shape[1]={K} (expected equal — t_obs is stored as time points)."
        )
    print(f"Reference: {args.reference_npz}")
    print(f"  N={ref['y0'].shape[0]}, K={K}, "
          f"t_obs[0..3]={t_points[:3]}, t_obs[-3..]={t_points[-3:]}")

    print(f"\nReading parquet pairs from {args.data_dir}")
    records: list[dict] = []
    n_skipped = 0
    for key, inp, out in _iter_parquet_pairs(args.data_dir):
        rec = _build_one_experiment(
            key, inp, out, t_points, K,
            mcherry_divisor=args.mcherry_divisor,
            outlier_mode=args.outlier_mode,
        )
        if rec is None:
            n_skipped += 1
            continue
        records.append(rec)
    print(f"  kept {len(records)} legacy experiments (skipped {n_skipped})")

    if not records:
        print("Nothing to write.")
        sys.exit(1)

    legacy = _build_combined_arrays(records, t_points)
    print(f"\nLegacy arrays:")
    print(f"  y0:    {legacy['y0'].shape}")
    print(f"  u_seq: {legacy['u_seq'].shape}")
    print(f"  y_seq: {legacy['y_seq'].shape}")
    print(f"  z_expr: all=1 (real)")

    if args.merge:
        print(f"\nMerging with {args.reference_npz} ...")
        out_dict = _concat_with_existing(legacy, args.reference_npz)
        n_real = int((out_dict["z_expr"] == 1).sum())
        n_synth = int((out_dict["z_expr"] == 0).sum())
        print(f"  merged N={out_dict['y0'].shape[0]}  (real={n_real}  synth={n_synth})")
    else:
        out_dict = legacy

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(args.output), **out_dict)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
