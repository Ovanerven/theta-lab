"""
Convert parsed real IVTT data (parquet files) into the .npz format
expected by last-layer-ode/train.py (ODEDataset).

This parser is aligned with the supervisor's `Oliver data parser.py`:
  - Reagent columns (everything except TIME and DNA) are MinMax-scaled
    jointly across all experiments.
  - `DNA c` is kept unscaled (raw concentration deltas) and collapsed
    into a single bolus at t=0 equal to the cumulative total. This
    reproduces supervisor's `dna_cum_total = cumsum(dna_raw)[:, -1, :]`
    assumption (DNA held constant at final concentration) while still
    using the existing `u_to_y_jump` mechanism to drive the DNA
    scaffold state.
  - y_seq observations: Broccoli (mm), mCherry/2 (pm).
  - y0 = obs at t0; y_seq[k] = obs at t_{k+1}; u_seq[k] = u during [t_k, t_{k+1}).

Usage:
    python datasets/convert_real_to_npz.py --layout full \
        --data-dir datasets/data_parsed_pruned \
        --output datasets/real_ivtt_full.npz

    python datasets/convert_real_to_npz.py --layout simple \
        --data-dir datasets/data_parsed_pruned \
        --output datasets/real_ivtt_simple.npz

Layout choices:
  full   : P=6 scaffold states [R, m, mm, p, pm, DNA] — matches
           TXTLMaturationDNAScaffold. Broccoli → mm (idx 2);
           mCherry/2 → pm (idx 4); DNA → idx 5.
  simple : P=3 scaffold states [mm, pm, DNA].
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


TIME_COL = "Time_seconds"
DNA_C_COL = "DNA c"
# Drop the raw per-step bolus column; DNA c is kept separately.
EXCLUDE_FROM_U = {TIME_COL, "DNA"}


LAYOUTS = {
    "full": {
        "P": 6,
        "state_names": ["R", "m", "mm", "p", "pm", "DNA"],
        "dna_idx": 5,
        "mm_idx": 2,
        "pm_idx": 4,
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
    parser.add_argument("--layout", type=str, default="full", choices=list(LAYOUTS.keys()))
    parser.add_argument("--data-dir", type=str, default="datasets/data_parsed_pruned")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--mcherry-divisor", type=float, default=2.0)
    parser.add_argument("--no-minmax", action="store_true",
                        help="Disable MinMax scaling of reagent columns (supervisor uses MinMax).")
    parser.add_argument(
        "--dna-mode",
        type=str,
        default="collapse",
        choices=["collapse", "stepwise"],
        help=(
            "DNA handling mode: 'collapse' routes total DNA c as a single t=0 bolus "
            "(supervisor-compatible), 'stepwise' keeps original per-step DNA c schedule."
        ),
    )
    parser.add_argument("--no-dna-collapse", action="store_true",
                        help="Deprecated alias for --dna-mode stepwise.")
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

    sample_inp = pd.read_parquet(pairs[0][0])
    control_cols = sorted([c for c in sample_inp.columns if c not in EXCLUDE_FROM_U])
    if DNA_C_COL not in control_cols:
        raise RuntimeError(
            f"Expected column '{DNA_C_COL}' to be kept in u_seq but it was excluded. "
            f"Columns seen: {list(sample_inp.columns)}"
        )
    dna_c_col_idx = control_cols.index(DNA_C_COL)
    d_in = len(control_cols)
    reagent_mask = np.ones(d_in, dtype=bool)
    reagent_mask[dna_c_col_idx] = False  # DNA c is not scaled

    print(f"Control columns ({d_in}): {control_cols}")
    dna_mode = args.dna_mode
    if args.no_dna_collapse:
        dna_mode = "stepwise"
        print("  NOTE: --no-dna-collapse is deprecated; use --dna-mode stepwise")
    collapse_dna = dna_mode == "collapse"

    print(f"  DNA c at u_seq idx {dna_c_col_idx} → routed to state '{state_names[dna_idx]}' (idx {dna_idx})")
    print(f"  minmax reagents: {not args.no_minmax} | DNA mode: {dna_mode}")

    expected_inp_cols = set(control_cols) | EXCLUDE_FROM_U

    # --- First pass: build per-experiment raw arrays (obs is independent of scaling) ---
    u_raw_list: list[np.ndarray] = []
    obs_list: list[np.ndarray] = []
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

        obs_full = np.stack([broccoli, mcherry], axis=1)
        # Match supervisor: drop last frame; u_seq[k] drives [t_k, t_{k+1}).
        u_full = df_inp[control_cols].to_numpy(dtype=np.float32)[:-1]

        y0_obs_list.append(obs_full[0])
        obs_list.append(obs_full[1:])
        u_raw_list.append(u_full)
        t_list.append(times)

    # --- Fit MinMaxScaler jointly on non-DNA-c reagent columns (supervisor behavior) ---
    if not args.no_minmax:
        all_reagents = np.concatenate([u[:, reagent_mask] for u in u_raw_list], axis=0)
        scaler = MinMaxScaler().fit(all_reagents)
        u_scale_min = scaler.data_min_.astype(np.float32)
        u_scale_max = scaler.data_max_.astype(np.float32)
        print(f"  MinMax fit over {all_reagents.shape[0]} steps, {all_reagents.shape[1]} reagents")
    else:
        scaler = None
        u_scale_min = np.zeros(int(reagent_mask.sum()), dtype=np.float32)
        u_scale_max = np.ones(int(reagent_mask.sum()), dtype=np.float32)

    # --- Apply scaling and DNA c collapse ---
    u_list: list[np.ndarray] = []
    dna_totals: list[float] = []
    for u_raw in u_raw_list:
        u = u_raw.copy()
        if scaler is not None:
            u[:, reagent_mask] = scaler.transform(u[:, reagent_mask]).astype(np.float32)
        dna_col = u[:, dna_c_col_idx].copy()
        dna_total = float(dna_col.sum())
        dna_totals.append(dna_total)
        if collapse_dna:
            u[:, dna_c_col_idx] = 0.0
            u[0, dna_c_col_idx] = dna_total  # single bolus at t=0
        u_list.append(u)

    dna_totals_arr = np.asarray(dna_totals, dtype=np.float32)
    print(f"  DNA totals (post-collapse): mean={dna_totals_arr.mean():.4g} "
          f"min={dna_totals_arr.min():.4g} max={dna_totals_arr.max():.4g}")

    # --- Determine padded dims ---
    lengths = np.array([o.shape[0] for o in obs_list], dtype=np.int64)
    K_max = int(lengths.max())
    N = len(obs_list)
    print(f"Samples: {N} | K range: [{lengths.min()}, {K_max}]")

    longest_idx = int(np.argmax(lengths))
    t_obs = t_list[longest_idx].astype(np.float32)

    # --- Build y0, y_seq, u_seq ---
    y0 = np.broadcast_to(x0_init, (N, P)).copy()
    y0[:, mm_idx] = np.asarray([y[0] for y in y0_obs_list], dtype=np.float32)
    y0[:, pm_idx] = np.asarray([y[1] for y in y0_obs_list], dtype=np.float32)

    y_seq = np.zeros((N, K_max, P), dtype=np.float32)
    u_seq = np.zeros((N, K_max, d_in), dtype=np.float32)

    for i in range(N):
        K_i = obs_list[i].shape[0]
        u_seq[i, :K_i] = u_list[i]
        y_seq[i, :K_i, mm_idx] = obs_list[i][:, 0]
        y_seq[i, :K_i, pm_idx] = obs_list[i][:, 1]
        if K_i < K_max:
            y_seq[i, K_i:, mm_idx] = obs_list[i][-1, 0]
            y_seq[i, K_i:, pm_idx] = obs_list[i][-1, 1]

    # --- u_to_y_jump routing (unchanged) ---
    obs_indices = np.arange(P, dtype=np.int64)
    control_indices = np.arange(P, P + d_in, dtype=np.int64)
    control_indices[dna_c_col_idx] = dna_idx

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
    reagent_col_names = np.array(
        [c for c, keep in zip(control_cols, reagent_mask) if keep], dtype="<U32"
    )

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
        # Scaler provenance for reproducibility / inverse transforms.
        u_scale_min=u_scale_min,
        u_scale_max=u_scale_max,
        u_scaled_cols=reagent_col_names,
        dna_totals=dna_totals_arr,
    )

    print(f"\nSaved to {out_path}")
    print(f"  y0:      {y0.shape}  (P={P})")
    print(f"  u_seq:   {u_seq.shape}")
    print(f"  y_seq:   {y_seq.shape}")
    print(f"  t_obs:   {t_obs.shape}")
    print(f"  lengths: {lengths.shape} (min={lengths.min()}, max={lengths.max()})")
    print(f"  state_names  : {list(obs_names)}")
    print(f"  obs_idx hint : [{mm_idx}, {pm_idx}]  (mm → Broccoli, pm → mCherry/2)")
    print(f"  dna_idx      : {dna_idx}  (t=0 bolus from u_seq col {dna_c_col_idx})")


if __name__ == "__main__":
    main()
