"""
Convert a Cantera-generated dataset from kmol/m³ to mole fractions in-place.
Writes a new .npz alongside the original with a _molfrac suffix.

Usage:
    python convert_to_mole_fractions.py datasets/aramco_kovacs14_fixed.npz
"""
import argparse
import numpy as np
from pathlib import Path


def to_mole_fractions(arr: np.ndarray) -> np.ndarray:
    """arr shape: (..., n_species). Divide each row by its sum."""
    c_total = arr.sum(axis=-1, keepdims=True).clip(1e-30)
    return arr / c_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=str)
    ap.add_argument("--out", type=str, default=None,
                    help="Output path. Defaults to <stem>_molfrac.npz")
    args = ap.parse_args()

    src = Path(args.dataset)
    dst = Path(args.out) if args.out else src.with_name(src.stem + "_molfrac.npz")

    d = np.load(str(src), allow_pickle=True)
    data = dict(d)

    # y0 and y_seq hold concentrations — convert both
    data["y0"]    = to_mole_fractions(data["y0"].astype(np.float32))
    data["y_seq"] = to_mole_fractions(data["y_seq"].astype(np.float32))
    # u_seq are bolus amounts (deltas) — leave as-is; they're control inputs not states

    np.savez(str(dst), **data)

    print(f"Saved to {dst}")
    print(f"y0    range: [{data['y0'].min():.4f}, {data['y0'].max():.4f}]")
    print(f"y_seq range: [{data['y_seq'].min():.4f}, {data['y_seq'].max():.4f}]")


if __name__ == "__main__":
    main()
