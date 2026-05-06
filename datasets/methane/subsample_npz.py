"""Subsample a trajectory dataset stored as an NPZ.

Designed for the `last-layer-ode/train.py` ODEDataset format:
  y0    : (N, P_obs)
  u_seq : (N, K, U)
  y_seq : (N, K, P_obs)
  t_obs : (K+1,)
  control_indices : (U,)
  obs_indices     : (P_obs,)

All keys whose first dimension equals N are subsampled; everything else is
copied through unchanged.

Example:
  python datasets/methane/subsample_npz.py \
    --source datasets/gri30_obs5_real_rates.npz \
    --out-dir datasets/methane \
    --name gri30_obs5 \
    --sizes 3,10,100,1000 \
    --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_sizes(s: str) -> list[int]:
    sizes: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        sizes.append(int(part))
    if not sizes:
        raise ValueError("--sizes must contain at least one integer")
    if any(n <= 0 for n in sizes):
        raise ValueError(f"--sizes must be positive; got {sizes}")
    return sizes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=str)
    ap.add_argument("--out-dir", default="datasets/methane", type=str)
    ap.add_argument("--name", default="gri30_obs5", type=str, help="Output base name (writes {name}_n{N}.npz)")
    ap.add_argument("--sizes", default="3,10,100,1000", type=str)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    sizes = _parse_sizes(args.sizes)
    sizes_sorted = sorted(set(sizes))

    src_path = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = np.load(str(src_path), allow_pickle=False)
    if "y0" not in src.files:
        raise ValueError(f"{src_path} missing required key 'y0'")

    N_full = int(src["y0"].shape[0])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N_full)

    for n in sizes_sorted:
        if n > N_full:
            raise ValueError(f"Requested n={n} but dataset only has N={N_full}")

        if n == N_full:
            idx = np.arange(N_full, dtype=np.int64)
        else:
            idx = np.sort(perm[:n].astype(np.int64))

        out: dict[str, np.ndarray] = {}
        for k in src.files:
            arr = src[k]
            if isinstance(arr, np.ndarray) and arr.shape != () and arr.shape[0] == N_full:
                out[k] = arr[idx]
            else:
                out[k] = arr

        out["subset_idx"] = idx
        out["subset_seed"] = np.array(args.seed, dtype=np.int64)
        out["subset_source"] = np.array(str(src_path), dtype="U")

        out_path = out_dir / f"{args.name}_n{n}.npz"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {out_path} (pass --overwrite)")

        np.savez(str(out_path), **out)
        print(f"Wrote {out_path}  (N={n})")


if __name__ == "__main__":
    main()
