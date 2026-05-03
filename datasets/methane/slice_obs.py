"""Slice an existing observation-subset npz into a smaller obs-subset npz.

The simulator was already run for the source dataset, so the trajectories are
bit-identical — we just drop columns from y0/y_seq and rewrite obs_indices/
obs_names. Everything else (controls, theta_true, t_obs, ...) carries over.

Usage:
    python datasets/methane/slice_obs.py \\
        --source datasets/gri30_obs5_real_rates.npz \\
        --keep-cols 0,1,2,3 \\
        --output datasets/gri30_obs4_westbrookdryer.npz
"""
import argparse
import numpy as np
from pathlib import Path


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=str)
    p.add_argument("--keep-cols", required=True, type=str,
                   help="Comma-separated column indices into the source obs axis (0..p_obs-1).")
    p.add_argument("--output", required=True, type=str)
    args = p.parse_args()

    keep = np.asarray(parse_int_list(args.keep_cols), dtype=np.int64)
    src = np.load(args.source, allow_pickle=False)

    p_obs_src = src["y_seq"].shape[-1]
    if keep.min() < 0 or keep.max() >= p_obs_src:
        raise ValueError(f"keep-cols {keep.tolist()} out of range for p_obs={p_obs_src}")

    out = {k: src[k] for k in src.files}
    out["y0"]          = src["y0"][:, keep]
    out["y_seq"]       = src["y_seq"][:, :, keep]
    out["obs_indices"] = src["obs_indices"][keep]
    out["obs_names"]   = src["obs_names"][keep]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **out)

    print(f"Sliced {args.source} -> {args.output}")
    print(f"  kept cols : {keep.tolist()}")
    print(f"  obs_names : {out['obs_names'].tolist()}")
    print(f"  obs_idx   : {out['obs_indices'].tolist()}")
    print(f"  y_seq     : {out['y_seq'].shape}")


if __name__ == "__main__":
    main()
