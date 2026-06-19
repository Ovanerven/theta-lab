"""Test script: run the same split logic train.py uses, no training.
Reports NEW vs OLD vs synth counts in each of train/val/test.

Usage:
    python scripts/check_split_balance.py [--n-seeds 3]

NEW vs OLD detection uses experiment_id prefix:
  - NEW = "REAL__2025-11-24_output__*" (Nov 2025 oxygen-aware experiments, 312 rows)
  - OLD = "YYYY-MM-DD_output.xlsx|||..." (Feb–May 2025 legacy, 691 rows)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Import the split function from train.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "last-layer-ode"))
from train import _make_split_indices  # noqa: E402


DATASET_PATH = "datasets/cell-free/txtl_combined_plus_legacy.npz"


def classify(experiment_id: np.ndarray, z_expr: np.ndarray) -> np.ndarray:
    """Return per-sample label: 'NEW', 'OLD', or 'synth'.
    NEW = experiment_id starts with 'REAL__' (Nov 2025 oxygen-aware experiments).
    OLD = everything else with z_expr==1 (legacy Feb–May 2025).
    """
    labels = np.empty(len(z_expr), dtype=object)
    labels[z_expr == 0] = "synth"
    ei_str = np.array([str(s) for s in experiment_id])
    new_mask = (z_expr == 1) & np.array([s.startswith("REAL__") for s in ei_str])
    old_mask = (z_expr == 1) & ~np.array([s.startswith("REAL__") for s in ei_str])
    labels[new_mask] = "NEW"
    labels[old_mask] = "OLD"
    return labels


def report(name: str, idx: np.ndarray, labels: np.ndarray) -> None:
    sub = labels[idx]
    n_new = int((sub == "NEW").sum())
    n_old = int((sub == "OLD").sum())
    n_synth = int((sub == "synth").sum())
    total = len(idx)
    n_real = n_new + n_old
    new_pct = 100 * n_new / max(1, n_real)
    print(
        f"  {name:<8} n={total:>5}  |  NEW={n_new:>4}  OLD={n_old:>4}  synth={n_synth:>5}"
        f"  |  NEW%-of-real={new_pct:5.1f}%"
    )


def run_one(z_expr: np.ndarray, lengths: np.ndarray, y_seq: np.ndarray,
            seed: int, mode: str, labels: np.ndarray) -> None:
    print(f"\n  seed={seed}  mode={mode}")
    n_val = n_test = 125 if "real_only" in mode else 1061
    test_real_only = "real_only" in mode
    stratified_split = "stratified" in mode
    stratify_z_expr = "z_expr" in mode

    train_idx, val_idx, test_idx = _make_split_indices(
        N=len(z_expr),
        y_seq=y_seq,
        lengths=lengths,
        n_val=n_val,
        n_test=n_test,
        split_seed=seed,
        stratified_split=stratified_split,
        stratify_bins=5,
        stratify_targets=[3, 5],
        z_expr=z_expr,
        stratify_z_expr=stratify_z_expr,
        test_real_only=test_real_only,
    )
    report("train", train_idx, labels)
    report("val",   val_idx,   labels)
    report("test",  test_idx,  labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=3)
    args = ap.parse_args()

    d = np.load(DATASET_PATH, allow_pickle=True)
    z_expr = d["z_expr"].astype(np.int64)
    lengths = d["lengths"].astype(np.int64)
    y_seq = d["y_seq"].astype(np.float32)
    experiment_id = d["experiment_id"]
    labels = classify(experiment_id, z_expr)

    n_total = len(z_expr)
    n_new = int((labels == "NEW").sum())
    n_old = int((labels == "OLD").sum())
    n_synth = int((labels == "synth").sum())
    print(f"Dataset: {DATASET_PATH}")
    print(f"  N total = {n_total}  ({n_new} NEW + {n_old} OLD + {n_synth} synth)")
    print(f"  NEW/(NEW+OLD) = {100*n_new/(n_new+n_old):.1f}%  (target real-split ratio)")

    modes = [
        "A4_current: stratify_z_expr (proportional synth in val/test) + stratified",
        "A4_new: test_real_only + stratified",
    ]

    for mode_label, mode_key in zip(modes, ["stratified+z_expr", "stratified+real_only"]):
        print(f"\n{'='*70}\n{mode_label}\n{'='*70}")
        for seed in range(1, args.n_seeds + 1):
            run_one(z_expr, lengths, y_seq, seed, mode_key, labels)


if __name__ == "__main__":
    main()
