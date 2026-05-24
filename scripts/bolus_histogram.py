"""Count bolus events per sample in a TXTL .npz dataset and print a histogram.

A "bolus event" at step k for sample b means |u_seq[b, k, :]|.sum() > 0
(same definition the V2 sparse-θ bolus-mode anchor mask uses).

Usage:
    python scripts/bolus_histogram.py datasets/cell-free/txtl_native_old_only.npz
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="Path to a TXTL .npz dataset")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="|u|.sum at a step must exceed this to count as a bolus (default 0.0)")
    args = ap.parse_args()

    d = np.load(args.npz)
    u = d["u_seq"]                                              # (N, T, U)
    lengths = d["lengths"] if "lengths" in d.files else None    # (N,) optional
    N, T, U = u.shape
    print(f"Dataset: {args.npz}")
    print(f"  N={N}  T={T}  U={U}")
    if lengths is not None:
        print(f"  lengths: min={int(lengths.min())} max={int(lengths.max())} "
              f"median={int(np.median(lengths))}")

    mag = np.abs(u).sum(axis=-1)                                # (N, T)
    bolus = mag > args.threshold                                # (N, T) bool

    # Per-sample bolus counts, respecting lengths if present.
    counts = np.empty(N, dtype=np.int64)
    for i in range(N):
        L = int(lengths[i]) if lengths is not None else T
        counts[i] = int(bolus[i, :L].sum())

    print()
    print(f"Bolus events per sample (threshold={args.threshold}):")
    print(f"  min={counts.min()}  max={counts.max()}  "
          f"mean={counts.mean():.2f}  median={int(np.median(counts))}  "
          f"p95={int(np.percentile(counts, 95))}  p99={int(np.percentile(counts, 99))}")
    print()

    hist = Counter(counts.tolist())
    print(f"  {'n_bolus':>8} | {'count':>6} | bar")
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*40}")
    max_count = max(hist.values())
    for n in sorted(hist.keys()):
        c = hist[n]
        bar = "#" * max(1, int(40 * c / max_count))
        print(f"  {n:>8} | {c:>6} | {bar}")

    # Cap-impact table: what fraction of samples are affected by various caps.
    print()
    print("Cap impact (how many samples would be truncated by max_anchors=k):")
    print(f"  {'cap':>4} | {'truncated':>9} | {'frac':>6}")
    print(f"  {'-'*4}-+-{'-'*9}-+-{'-'*6}")
    for cap in (1, 2, 3, 4, 6, 8, 12, 16, 24, 48):
        n_trunc = int((counts > cap).sum())
        frac = n_trunc / N
        print(f"  {cap:>4} | {n_trunc:>9} | {frac:>6.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
