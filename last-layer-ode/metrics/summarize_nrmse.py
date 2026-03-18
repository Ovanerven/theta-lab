"""Scan an experiment folder, evaluate all scaffolds on their test split, write NRMSE CSVs."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plot_diagnostics import rebuild_model_from_experiment, device_auto
from scaffolds import SCAFFOLDS


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    rng = float(np.max(y_true) - np.min(y_true))
    return rmse / max(rng, 1e-8)


def get_prediction(model, ds, idx, device):
    y0, u_seq, y_seq = ds[idx]
    dt = torch.tensor(ds.dt.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(
            y0.unsqueeze(0).to(device),
            u_seq.unsqueeze(0).to(device),
            dt,
            y_seq=None,
            teacher_forcing=False,
        )
    pred = out[0] if isinstance(out, (tuple, list)) else out
    return y_seq.cpu().numpy(), pred[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_root", type=str)
    parser.add_argument("--species", nargs="*", default=["A", "M"])
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--drop-diverged", type=float, default=None, metavar="THRESH",
                        help="Drop trajectories where any species NRMSE exceeds this threshold "
                             "in any scaffold (e.g. 100). Dropped consistently across all scaffolds.")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    species = [s.upper() for s in args.species]
    device = device_auto()

    # find all experiment dirs
    exp_dirs = sorted(
        d for d in exp_root.iterdir()
        if d.is_dir() and (d / "config.yaml").exists() and (d / "model.pt").exists()
    )
    print(f"Found {len(exp_dirs)} experiments in {exp_root}")

    rows = []
    for exp_dir in exp_dirs:
        cfg = yaml.safe_load((exp_dir / "config.yaml").read_text())
        scaffold = cfg.get("scaffold", exp_dir.name)
        sc_key = scaffold if scaffold in SCAFFOLDS else None
        P = SCAFFOLDS[sc_key].P if sc_key else 0

        # load test indices from split.npz
        split_path = exp_dir / "split.npz"
        if not split_path.exists():
            print(f"  skip {scaffold}: no split.npz")
            continue
        test_idx = np.load(split_path)["test_idx"]
        if len(test_idx) == 0:
            print(f"  skip {scaffold}: empty test set")
            continue

        model, ds, state_names, _ = rebuild_model_from_experiment(exp_dir, device)
        snames = [s.upper() for s in state_names]

        missing = [s for s in species if s not in snames]
        if missing:
            print(f"  skip {scaffold}: missing {missing}")
            continue

        print(f"  {scaffold} (P={P}) — {len(test_idx)} test trajectories")
        for idx in test_idx:
            if idx >= len(ds):
                continue
            y_true, y_pred = get_prediction(model, ds, int(idx), device)
            for sp in species:
                j = snames.index(sp)
                v = nrmse(y_true[:, j], y_pred[:, j])
                rows.append({
                    "scaffold": scaffold,
                    "P": P,
                    "sample_idx": int(idx),
                    "species": sp,
                    "nrmse": v,
                })

    # detailed CSV
    out_path = Path(args.out) if args.out else exp_root / "nrmse_detailed.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scaffold", "P", "sample_idx", "species", "nrmse"])
        w.writeheader()
        w.writerows(rows)

    # summary CSV
    df = pd.DataFrame(rows)

    # drop numerically diverged trajectories (consistently across all scaffolds)
    if args.drop_diverged is not None:
        thresh = args.drop_diverged
        diverged_idx = set(df[df["nrmse"] > thresh]["sample_idx"].unique())
        if diverged_idx:
            n_before = df["sample_idx"].nunique()
            df = df[~df["sample_idx"].isin(diverged_idx)]
            print(f"\n  Dropped {len(diverged_idx)}/{n_before} trajectories with NRMSE > {thresh} "
                  f"in any scaffold/species. Keeping {df['sample_idx'].nunique()}.")

    agg = df.groupby(["scaffold", "P", "species"])["nrmse"].agg(
        ["count", "mean", "median", "std"]
    ).reset_index()
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    # count trajectories where NRMSE > 1 (exploded predictions)
    exploded = df[df["nrmse"] > 1.0].groupby(["scaffold", "P", "species"]).size().reset_index(name="n_exploded")
    agg = agg.merge(exploded, on=["scaffold", "P", "species"], how="left")
    agg["n_exploded"] = agg["n_exploded"].fillna(0).astype(int)
    agg_path = out_path.with_name(out_path.stem.replace("detailed", "summary") + ".csv")
    agg.to_csv(agg_path, index=False)

    # print warning for scaffolds with explosions
    bad = agg[agg["n_exploded"] > 0]
    if len(bad) > 0:
        print(f"\n⚠ {len(bad)} scaffold/species combos have exploding trajectories (NRMSE > 1):")
        for _, r in bad.iterrows():
            print(f"    {r['scaffold']} {r['species']}: {r['n_exploded']}/{r['count']} exploded, "
                  f"mean={r['mean']:.2f}, median={r['median']:.4f}")

    print(f"\nSaved:\n  {out_path}\n  {agg_path}")


if __name__ == "__main__":
    main()