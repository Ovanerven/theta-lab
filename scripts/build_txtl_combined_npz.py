"""Build a combined real + synthetic-no-go TXTL dataset .npz from raw Excel files.

Pipeline:
    1. Read one or more real-experiment Excel workbooks from --real-dir
       (each sheet = one experiment, cumulative-nL inputs).
    2. Generate synthetic no-go experiments per real workbook via
       last-layer-ode/sim/txtl_synthetic_no_go.py.
    3. Convert cumulative-nL input columns to per-step deltas.
    4. Compute a `DNA c` concentration column following bob_model/parse_IVTT_data.py
       `_compute_DNA`: DNA_cumsum / total_volume_cumsum * dna_conc_multiplier,
       then per-step diff.
    5. Pack everything into a single .npz with the same field layout as
       scripts/convert_real_to_npz.py (so the existing trainers and scaffolds
       work unchanged), plus a new `z_expr` flag column (1 for real, 0 for
       synthetic no-go) and a `class_label` string column.

Output layout (matches `--layout full` in convert_real_to_npz.py):
    state_names    = ['R', 'O', 'm', 'mm', 'p', 'pm', 'DNA']           (P=7)
    control_names  = ['3PGA','AA','DNA c','FB','K-Glut','Lysate 2%PEG',
                       'Maltose','Mg-Glut','NTPs','PEG8000','T7RNAP','water']  (d_in=12)
    DNA c idx in u_seq -> state idx 6 (DNA) via control_indices.

Usage:
    python scripts/build_txtl_combined_npz.py \\
        --real-dir next_steps/real_data \\
        --output datasets/cell-free/txtl_combined.npz \\
        --dna-conc-multiplier 30.0 \\
        --mode randomized --n-random-per-template 3 --seed 7

Notes:
    - The DNA concentration multiplier mirrors Bob's per-group `conc_series['DNA']`
      value. There is no group_files_dict on this machine, so the multiplier is a
      CLI arg. Use --inspect-existing path/to/old.npz to read it off an existing
      build.
    - Excel sheets without all required columns are skipped with a warning.
    - mCherry [RFU] is divided by --mcherry-divisor (default 2.0) before being
      written as the `pm` state value, matching convert_real_to_npz.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow `import sim.txtl_synthetic_no_go` from the repo's last-layer-ode package
# regardless of where this script is invoked from.
REPO_ROOT = Path(__file__).resolve().parents[1]
LLO_ROOT = REPO_ROOT / "last-layer-ode"
if str(LLO_ROOT) not in sys.path:
    sys.path.insert(0, str(LLO_ROOT))

from sim.txtl_synthetic_no_go import (  # noqa: E402
    INPUT_COLS as RAW_INPUT_COLS,
    TIME_COL,
    MRNA_RFU_COL,
    PROTEIN_RFU_COL,
    read_workbook,
    generate_synthetic,
)


# Target full schema (matches scripts/convert_real_to_npz.py "full" layout).
STATE_NAMES = ["R", "O", "m", "mm", "p", "pm", "DNA"]
P = 7
DNA_IDX = 6
MM_IDX = 3
PM_IDX = 5
X0_INIT = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# 12-column control schema. RAW_INPUT_COLS already has all 12 cumulative-nL
# columns (3PGA, AA, DNA, FB, K-Glut, Lysate 2%PEG, Maltose, Mg-Glut, NTPs,
# PEG8000, T7RNAP, water). After diffing, DNA is replaced with a `DNA c`
# concentration column.
EXTRA_INPUT_COLS: list[str] = []  # kept for back-compat; all reagents now live in RAW_INPUT_COLS
CONTROL_NAMES = sorted([c for c in RAW_INPUT_COLS if c != "DNA"] + ["DNA c"])
assert "DNA" not in CONTROL_NAMES
DNA_C_IDX = CONTROL_NAMES.index("DNA c")
D_IN = len(CONTROL_NAMES)

PM_OUTLIER_THRESHOLD = 5000.0


def _per_step_deltas(cumulative: np.ndarray) -> np.ndarray:
    """Cumulative-nL -> per-step deltas; first row keeps the initial level."""
    deltas = np.diff(cumulative, axis=0, prepend=0.0)
    return deltas


def _compute_dna_c(cumulative_nL_df: pd.DataFrame, dna_conc_multiplier: float) -> np.ndarray:
    """Mirror bob_model/parse_IVTT_data._compute_DNA followed by .diff().

    DNA c(t) = (cumsum_DNA(t) / cumsum_total(t)) * multiplier, then per-step diff.
    The result is the per-step *concentration delta* of DNA in the well, which is
    what the existing u_to_y_jump routing expects on the DNA c column.
    """
    df = cumulative_nL_df[RAW_INPUT_COLS].astype(float).reset_index(drop=True)
    cumsum_total = df.sum(axis=1).replace(0.0, np.nan)
    frac = df["DNA"] / cumsum_total
    conc = (frac * float(dna_conc_multiplier)).fillna(0.0).to_numpy(dtype=np.float64)
    delta_conc = np.diff(conc, prepend=0.0)
    delta_conc[~np.isfinite(delta_conc)] = 0.0
    return delta_conc.astype(np.float32)


def _build_u_seq_row(
    df: pd.DataFrame,
    dna_conc_multiplier: float,
) -> np.ndarray:
    """Build one experiment's u_seq row in the 12-col CONTROL_NAMES schema.

    Per-step deltas of every nL input column (DNA omitted), plus a DNA c
    concentration delta column, plus zero-filled 3PGA/AA/NTPs (absent in this
    Excel format).
    """
    n = len(df)
    out = np.zeros((n, D_IN), dtype=np.float32)

    # Per-step nL deltas for the columns that are physically pipetted in.
    for col in RAW_INPUT_COLS:
        if col == "DNA":
            continue
        idx = CONTROL_NAMES.index(col)
        cum = df[col].astype(float).to_numpy()
        out[:, idx] = _per_step_deltas(cum)

    # DNA c concentration delta.
    out[:, DNA_C_IDX] = _compute_dna_c(df, dna_conc_multiplier)

    return out


def _build_y0_y_seq(df: pd.DataFrame, mcherry_divisor: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y0[P], y_seq[K,P]) in the 7-state full layout.

    Only mm (Broccoli) and pm (mCherry/divisor) are filled from observations;
    other states default to X0_INIT.
    """
    n = len(df)
    broccoli = df[MRNA_RFU_COL].astype(float).to_numpy(dtype=np.float32)
    mcherry = df[PROTEIN_RFU_COL].astype(float).to_numpy(dtype=np.float32) / float(mcherry_divisor)

    y0 = X0_INIT.copy()
    y0[MM_IDX] = float(broccoli[0])
    y0[PM_IDX] = float(mcherry[0])

    y_seq = np.zeros((n, P), dtype=np.float32)
    y_seq[:, MM_IDX] = broccoli
    y_seq[:, PM_IDX] = mcherry
    return y0, y_seq


def _drop_pm_outlier(pm_values: np.ndarray, mode: str) -> bool:
    if mode == "none":
        return False
    if mode == "max":
        return float(np.nanmax(pm_values)) > PM_OUTLIER_THRESHOLD
    if mode == "last":
        return float(pm_values[-1]) > PM_OUTLIER_THRESHOLD
    raise ValueError(f"Unknown outlier-mode: {mode!r}")


def _iter_real_sheets(real_dir: Path):
    """Yield (workbook_path, sheet_name, df) for every readable sheet in real_dir."""
    real_dir = Path(real_dir)
    xlsx_files = sorted(real_dir.glob("*.xlsx")) + sorted(real_dir.glob("*.xls"))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx/.xls files found under {real_dir}")

    required = set(RAW_INPUT_COLS + [TIME_COL, MRNA_RFU_COL, PROTEIN_RFU_COL])

    for xp in xlsx_files:
        xls = pd.ExcelFile(xp)
        for sheet in xls.sheet_names:
            df = pd.read_excel(xp, sheet_name=sheet)
            df.columns = [str(c).strip() for c in df.columns]
            missing = required.difference(df.columns)
            non_recoverable = [c for c in missing if c == TIME_COL or c in (MRNA_RFU_COL, PROTEIN_RFU_COL)]
            if non_recoverable:
                print(f"  skip {xp.name}!{sheet}: missing non-recoverable cols {sorted(non_recoverable)}")
                continue
            if missing:
                print(f"  backfill {xp.name}!{sheet}: missing reagent cols {sorted(missing)} → 0")
                for c in missing:
                    df[c] = 0.0
            for c in RAW_INPUT_COLS:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
            df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
            for c in (MRNA_RFU_COL, PROTEIN_RFU_COL):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            yield xp, sheet, df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real-dir", type=Path, required=True,
                        help="Folder containing real-experiment Excel workbooks.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output .npz path.")
    parser.add_argument("--dna-conc-multiplier", type=float, default=30.0,
                        help="DNA stock concentration multiplier for `DNA c` "
                             "(mirrors Bob's per-group conc_series['DNA']). Default 30.0.")
    parser.add_argument("--mcherry-divisor", type=float, default=2.0,
                        help="Divide raw mCherry RFU by this for the pm state. Default 2.0.")
    parser.add_argument("--outlier-mode", type=str, default="last",
                        choices=["max", "last", "none"],
                        help="Drop experiments whose pm exceeds 5000 RFU. Default 'last' (bob_model-style).")
    parser.add_argument("--mode", type=str, default="randomized",
                        choices=["deterministic", "randomized", "both"],
                        help="Synthetic generation mode (passed to the sim).")
    parser.add_argument("--n-random-per-template", type=int, default=3,
                        help="Randomized replicates per template real-sheet (per rule).")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--include-synthetic", action="store_true", default=True,
                        help="Include synthetic no-go experiments (default: on).")
    parser.add_argument("--no-synthetic", dest="include_synthetic", action="store_false",
                        help="Only build the real-data half.")
    args = parser.parse_args()

    real_dir: Path = args.real_dir
    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading real workbooks from: {real_dir}")
    real_records: List[Dict] = []
    n_dropped_real = 0
    for wb_path, sheet, df in _iter_real_sheets(real_dir):
        pm = df[PROTEIN_RFU_COL].astype(float).to_numpy() / float(args.mcherry_divisor)
        if _drop_pm_outlier(pm, args.outlier_mode):
            n_dropped_real += 1
            print(f"  drop outlier {wb_path.name}!{sheet} (pm exceeds {PM_OUTLIER_THRESHOLD:.0f})")
            continue
        u_row = _build_u_seq_row(df, args.dna_conc_multiplier)
        y0, y_row = _build_y0_y_seq(df, args.mcherry_divisor)
        t = df[TIME_COL].astype(np.float32).to_numpy()
        real_records.append({
            "experiment_id": f"REAL__{wb_path.stem}__{sheet}",
            "z_expr": 1,
            "class_label": "real",
            "u": u_row,
            "y0": y0,
            "y_seq": y_row,
            "t": t,
            "dna_total": float(df["DNA"].astype(float).iloc[-1]),
        })

    print(f"  kept {len(real_records)} real experiments (dropped {n_dropped_real} pm-outliers)")

    synth_records: List[Dict] = []
    if args.include_synthetic:
        print(f"Generating synthetic no-go experiments per real workbook...")
        workbook_paths = sorted(set(Path(real_dir).glob("*.xlsx")) | set(Path(real_dir).glob("*.xls")))
        for wb_path in workbook_paths:
            wb = read_workbook(wb_path)
            experiments, _meta, _summary = generate_synthetic(
                wb, seed=args.seed, mode=args.mode,
                n_random_per_template=args.n_random_per_template,
            )
            for exp in experiments:
                df = exp.dataframe.copy()
                df.columns = [str(c).strip() for c in df.columns]
                for c in RAW_INPUT_COLS:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
                df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
                for c in (MRNA_RFU_COL, PROTEIN_RFU_COL):
                    df[c] = pd.to_numeric(df[c], errors="coerce")

                u_row = _build_u_seq_row(df, args.dna_conc_multiplier)
                y0, y_row = _build_y0_y_seq(df, args.mcherry_divisor)
                t = df[TIME_COL].astype(np.float32).to_numpy()
                synth_records.append({
                    "experiment_id": f"SYNTH__{wb_path.stem}__{exp.sheet_name}",
                    "z_expr": 0,
                    "class_label": str(exp.metadata.get("class_label", "synthetic_no_go")),
                    "u": u_row,
                    "y0": y0,
                    "y_seq": y_row,
                    "t": t,
                    "dna_total": float(df["DNA"].astype(float).iloc[-1]),
                })

        print(f"  generated {len(synth_records)} synthetic experiments")

    records = real_records + synth_records
    if not records:
        raise RuntimeError("No experiments collected; aborting.")

    # Pad to common length.
    lengths = np.array([r["u"].shape[0] for r in records], dtype=np.int64)
    K_max = int(lengths.max())
    N = len(records)

    longest_idx = int(np.argmax(lengths))
    t_obs = records[longest_idx]["t"].astype(np.float32)
    # Dataset contract: t_obs has length K+1 (interval endpoints), u_seq/y_seq
    # have length K (one per interval). Raw records carry one entry per
    # observation timepoint, so K_intervals = K_max - 1.
    K_intervals = K_max - 1

    y0 = np.zeros((N, P), dtype=np.float32)
    y_seq = np.zeros((N, K_intervals, P), dtype=np.float32)
    u_seq = np.zeros((N, K_intervals, D_IN), dtype=np.float32)
    z_expr = np.zeros((N,), dtype=np.int64)
    class_label = np.zeros((N,), dtype="<U32")
    experiment_id = np.zeros((N,), dtype="<U96")
    dna_totals = np.zeros((N,), dtype=np.float32)

    for i, r in enumerate(records):
        K_i = int(r["u"].shape[0]) - 1  # intervals for this row
        y0[i] = r["y0"]
        # r["u"][0] is the t=0 loading (DNA c is bolus-at-t=0 by definition,
        # and other reagents that were pipetted in before the run also land
        # here via _per_step_deltas with prepend=0.0). Fold it into the first
        # interval so the model's first ODE step sees it.
        u_int = r["u"].copy()
        u_int[1] += u_int[0]
        u_seq[i, :K_i] = u_int[1:]
        y_seq[i, :K_i] = r["y_seq"][1:]
        if K_i < K_intervals:
            y_seq[i, K_i:] = r["y_seq"][-1]
        z_expr[i] = r["z_expr"]
        class_label[i] = r["class_label"][:32]
        experiment_id[i] = r["experiment_id"][:96]
        dna_totals[i] = r["dna_total"]

    lengths = lengths - 1

    # u_to_y_jump routing: DNA c column -> DNA state.
    obs_indices = np.arange(P, dtype=np.int64)
    control_indices = np.arange(P, P + D_IN, dtype=np.int64)
    control_indices[DNA_C_IDX] = DNA_IDX

    # y_seq raw stats (matches convert_real_to_npz.py output for downstream norm).
    obs_raw_min = y_seq.reshape(-1, P).min(axis=0).astype(np.float32)
    obs_raw_max = y_seq.reshape(-1, P).max(axis=0).astype(np.float32)
    valid_2d = (np.arange(K_intervals)[None, :] < lengths[:, None]).astype(np.float32)
    valid_sum = float(valid_2d.sum())
    obs_raw_mean = (y_seq * valid_2d[..., None]).sum(axis=(0, 1)) / max(valid_sum, 1.0)
    obs_raw_var = ((y_seq - obs_raw_mean) ** 2 * valid_2d[..., None]).sum(axis=(0, 1)) / max(valid_sum, 1.0)
    obs_raw_std = np.sqrt(np.maximum(obs_raw_var, 1e-12)).astype(np.float32)

    np.savez(
        out_path,
        y0=y0,
        u_seq=u_seq,
        y_seq=y_seq,
        t_obs=t_obs,
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
        outlier_mode=np.array(args.outlier_mode, dtype="<U16"),
        dna_totals=dna_totals,
        obs_raw_min=obs_raw_min,
        obs_raw_max=obs_raw_max,
        obs_raw_mean=obs_raw_mean.astype(np.float32),
        obs_raw_std=obs_raw_std.astype(np.float32),
        # new fields
        z_expr=z_expr,
        class_label=class_label,
        experiment_id=experiment_id,
        dna_conc_multiplier=np.float32(args.dna_conc_multiplier),
    )

    n_real = int((z_expr == 1).sum())
    n_synth = int((z_expr == 0).sum())
    print(f"\nWrote {out_path}")
    print(f"  N total = {N}  (real={n_real}, synthetic no-go={n_synth})")
    print(f"  u_seq   = {u_seq.shape}   control_names={list(CONTROL_NAMES)}")
    print(f"  y_seq   = {y_seq.shape}   state_names={STATE_NAMES}")
    print(f"  DNA c routes u_seq[:, {DNA_C_IDX}]  ->  state idx {DNA_IDX} ({STATE_NAMES[DNA_IDX]})")


if __name__ == "__main__":
    main()
