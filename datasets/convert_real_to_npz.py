"""
Convert parsed real IVTT data (parquet files) into the .npz format
expected by last-layer-ode/train.py (ODEDataset).

Usage:
    python datasets/convert_real_to_npz.py --layout full \
        --data-dir datasets/data_parsed_pruned \
        --output datasets/real_ivtt_maturation.npz

    python datasets/convert_real_to_npz.py --layout simple \
        --data-dir datasets/data_parsed_pruned \
        --output datasets/real_ivtt_simple.npz

Layout choices:
  full   : P=6 scaffold states [R, m, mm, p, pm, DNA] — matches
           TXTLMaturationDNAScaffold. Broccoli → state index 2 (mm);
           mCherry/2 → state index 4 (pm).
  simple : P=3 scaffold states [mm, pm, DNA] — matches
           TXTLSimpleDNAScaffold. Broccoli → state 0; mCherry/2 → state 1.

DNA handling
------------
The raw per-step bolus column is named "DNA" (zero almost everywhere, except
the single pipette moment). The column "DNA c" is the supervisor's
dilution-corrected concentration delta: cumsum("DNA c") over time equals the
actual [DNA](t) concentration including volume tracking.

This converter EXCLUDES the raw "DNA" column from u_seq. "DNA c" is kept in
u_seq and routed via u_to_y_jump onto the DNA scaffold state, so that each
step applies y[..., DNA_idx] += DNA_c_k. Because the scaffold sets
dDNA/dt = 0 between jumps, the DNA state at step k is exactly cumsum(DNA c)
up to k — no manual cumulative construction needed.

The 11 other reagents stay as non-state-modifying controls: they appear in
u_seq (so the encoder sees them) but their column entries in u_to_y_jump
are zero.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


TIME_COL = "Time_seconds"
# Drop the raw bolus column; keep "DNA c" (the concentration delta) in u_seq.
EXCLUDE_FROM_U = {TIME_COL, "DNA"}
DNA_C_COL = "DNA c"


# Scaffold layouts. Each entry describes:
#   P            : scaffold state dim
#   state_names  : ordered state list (for obs_names)
#   dna_idx      : index of the DNA state inside the P states
#   mm_idx       : index of the observed mRNA / Broccoli state
#   pm_idx       : index of the observed mature-protein / mCherry state
#   x0_init      : length-P initial state vector template
LAYOUTS = {
    "full": {
        "P": 6,
        "state_names": ["R", "m", "mm", "p", "pm", "DNA"],
        "dna_idx": 5,
        "mm_idx": 2,
        "pm_idx": 4,
        # R starts at 1.0 (fresh resource pool). Everything else at 0.
        "x0_init": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    "simple": {
        "P": 3,
        "state_names": ["mm", "pm", "DNA"],
        "dna_idx": 2,
        "mm_idx": 0,
        "pm_idx": 1,
        "x0_init": [0.0, 0.0, 0.0],
    },
}


def collect_experiment_pairs(data_dir: str) -> list[tuple[str, str]]:
    inp_files = sorted(glob.glob(os.path.join(data_dir, "*__inputs.parquet")))
    pairs = []
    for inp_path in inp_files:
        out_path = inp_path.replace("__inputs.parquet", "__outputs.parquet")
        if os.path.exists(out_path):
            pairs.append((inp_path, out_path))
        else:
            print(f"WARNING: no matching outputs for {inp_path}, skipping")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Convert real IVTT parquet data to .npz")
    parser.add_argument("--layout", type=str, default="full", choices=list(LAYOUTS.keys()),
                        help="Scaffold layout: 'full' (P=6) or 'simple' (P=3)")
    parser.add_argument("--data-dir", type=str, default="datasets/data_parsed_pruned")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npz path. Defaults to datasets/real_ivtt_<layout>.npz")
    parser.add_argument("--mcherry-divisor", type=float, default=2.0,
                        help="Divide mCherry RFU by this value (default 2.0)")
    args = parser.parse_args()

    layout = LAYOUTS[args.layout]
    P = layout["P"]
    state_names = layout["state_names"]
    dna_idx = layout["dna_idx"]
    mm_idx = layout["mm_idx"]
    pm_idx = layout["pm_idx"]
    x0_init = np.asarray(layout["x0_init"], dtype=np.float32)

    output = args.output or f"datasets/real_ivtt_{args.layout}.npz"

    pairs = collect_experiment_pairs(args.data_dir)
    if not pairs:
        raise RuntimeError(f"No experiment pairs found in {args.data_dir}")
    print(f"Found {len(pairs)} experiments | layout={args.layout} (P={P})")

    # --- Determine control columns from first file ---
    sample_inp = pd.read_parquet(pairs[0][0])
    control_cols = sorted([c for c in sample_inp.columns if c not in EXCLUDE_FROM_U])
    if DNA_C_COL not in control_cols:
        raise RuntimeError(
            f"Expected column '{DNA_C_COL}' to be kept in u_seq but it was excluded. "
            f"Columns seen: {list(sample_inp.columns)}"
        )
    dna_c_col_idx = control_cols.index(DNA_C_COL)
    d_in = len(control_cols)

    print(f"Control columns ({d_in}): {control_cols}")
    print(f"  DNA c column is at u_seq index {dna_c_col_idx} → routed to state '{state_names[dna_idx]}' (idx {dna_idx})")

    expected_inp_cols = set(control_cols) | EXCLUDE_FROM_U

    # --- Collect per-experiment arrays ---
    u_list: list[np.ndarray] = []
    obs_list: list[np.ndarray] = []   # observed (broccoli, mcherry/2) per step
    y0_obs_list: list[np.ndarray] = []
    t_list: list[np.ndarray] = []

    for inp_path, out_path in pairs:
        df_inp = pd.read_parquet(inp_path)
        assert set(df_inp.columns) == expected_inp_cols, (
            f"Column mismatch in {inp_path}: "
            f"extra={set(df_inp.columns) - expected_inp_cols}, "
            f"missing={expected_inp_cols - set(df_inp.columns)}"
        )
        df_out = pd.read_parquet(out_path)

        times = df_out[TIME_COL].to_numpy(dtype=np.float32)
        broccoli = df_out["Broccoli [RFU]"].to_numpy(dtype=np.float32)
        mcherry = df_out["mCherry [RFU]"].to_numpy(dtype=np.float32) / args.mcherry_divisor

        obs_full = np.stack([broccoli, mcherry], axis=1)  # (T, 2)
        u_full = df_inp[control_cols].to_numpy(dtype=np.float32)  # (T, U)

        # y0 = first observation; y_seq[k] = observation at t_{k+1}
        # u_seq[k] = input during interval [t_k, t_{k+1})
        y0_obs_list.append(obs_full[0])
        obs_list.append(obs_full[1:])
        u_list.append(u_full[:-1])
        t_list.append(times)

    # --- Determine dimensions and pad ---
    lengths = np.array([o.shape[0] for o in obs_list], dtype=np.int64)
    K_max = int(lengths.max())
    N = len(obs_list)

    print(f"Samples: {N} | K range: [{lengths.min()}, {K_max}]")

    # Use longest experiment's time grid as the common t_obs
    longest_idx = int(np.argmax(lengths))
    t_obs = t_list[longest_idx].astype(np.float32)  # (K_max+1,)

    # --- Build y0 and y_seq in scaffold-state layout ---
    # y0 shape (N, P): template x0, with Broccoli/mCherry pasted into mm/pm
    y0 = np.broadcast_to(x0_init, (N, P)).copy()
    y0[:, mm_idx] = np.asarray([y[0] for y in y0_obs_list], dtype=np.float32)
    y0[:, pm_idx] = np.asarray([y[1] for y in y0_obs_list], dtype=np.float32)

    # y_seq shape (N, K_max, P). Hidden states (all slots other than mm/pm)
    # stay zero; the training loop will ignore them because obs_idx masks them.
    y_seq = np.zeros((N, K_max, P), dtype=np.float32)
    u_seq = np.zeros((N, K_max, d_in), dtype=np.float32)

    for i in range(N):
        K_i = obs_list[i].shape[0]
        u_seq[i, :K_i] = u_list[i]
        y_seq[i, :K_i, mm_idx] = obs_list[i][:, 0]
        y_seq[i, :K_i, pm_idx] = obs_list[i][:, 1]
        # Forward-fill observed channels in padded region (hold last value),
        # so per-batch loss masking (via `lengths`) is the only thing gating
        # the loss — but the forward-fill keeps values sane if something leaks.
        if K_i < K_max:
            y_seq[i, K_i:, mm_idx] = obs_list[i][-1, 0]
            y_seq[i, K_i:, pm_idx] = obs_list[i][-1, 1]

    # --- Build u_to_y_jump routing ---
    # The jump matrix is built by `make_u_to_y_jump(control_indices, obs_indices)`
    # which sets J[j, p] = 1 iff control_indices[j] == obs_indices[p] in a shared
    # integer namespace. We want ONLY (j = dna_c_col_idx, p = dna_idx) to fire.
    #
    # Scheme:
    #   - obs_indices occupy slots [0 .. P-1].
    #   - control_indices use unique tags >= P for inert columns, except the
    #     DNA c column, which shares the DNA state's tag (= dna_idx).
    obs_indices = np.arange(P, dtype=np.int64)
    control_indices = np.arange(P, P + d_in, dtype=np.int64)
    control_indices[dna_c_col_idx] = dna_idx  # route DNA c onto DNA state

    # Defensive check: after the override, exactly one control index should
    # match exactly one obs index (the DNA routing).
    matches = [
        (j, p) for j in range(d_in) for p in range(P)
        if int(control_indices[j]) == int(obs_indices[p])
    ]
    assert matches == [(dna_c_col_idx, dna_idx)], (
        f"Expected exactly one routing (j={dna_c_col_idx}, p={dna_idx}), got {matches}"
    )

    control_names = np.array(control_cols, dtype="<U32")
    obs_names = np.array(state_names, dtype="<U32")
    names_full = np.array(state_names + control_cols, dtype="<U32")

    # --- Save ---
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(out_path),
        y0=y0.astype(np.float32),
        u_seq=u_seq,
        y_seq=y_seq,
        t_obs=t_obs,
        control_indices=control_indices,
        obs_indices=obs_indices,
        names_full=names_full,
        control_names=control_names,
        obs_names=obs_names,
        n_states_full=np.int64(P),
        n_params_full=np.int64(0),
        theta_true=np.zeros(0, dtype=np.float32),
        lengths=lengths,
    )

    print(f"\nSaved to {out_path}")
    print(f"  y0:      {y0.shape}  (P={P})")
    print(f"  u_seq:   {u_seq.shape}")
    print(f"  y_seq:   {y_seq.shape}")
    print(f"  t_obs:   {t_obs.shape}")
    print(f"  lengths: {lengths.shape} (min={lengths.min()}, max={lengths.max()})")
    print(f"  state_names  : {list(obs_names)}")
    print(f"  obs_idx hint : [{mm_idx}, {pm_idx}]  (mm → Broccoli, pm → mCherry/2)")
    print(f"  dna_idx      : {dna_idx}  (routed via u_to_y_jump from u_seq col {dna_c_col_idx})")


if __name__ == "__main__":
    main()
