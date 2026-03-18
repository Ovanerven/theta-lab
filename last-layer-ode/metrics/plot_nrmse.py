"""Read an NRMSE summary or detailed CSV and plot NRMSE vs P."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str, help="nrmse_detailed.csv or nrmse_summary.csv")
    parser.add_argument("--stat", choices=["mean", "median"], default="median")
    parser.add_argument("--error-bar", choices=["sem", "std", "iqr"], default="iqr")
    parser.add_argument("--cap", type=float, default=None,
                        help="Cap NRMSE values at this threshold before aggregating (e.g. 1.0)")
    parser.add_argument("--drop-exploded", action="store_true",
                        help="Drop trajectories with NRMSE > --cap instead of capping")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--format", choices=["pdf", "png"], default="pdf")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    # if detailed CSV, optionally cap/drop, then aggregate
    if "sample_idx" in df.columns:
        if args.cap is not None:
            n_before = len(df)
            if args.drop_exploded:
                df = df[df["nrmse"] <= args.cap]
                print(f"Dropped {n_before - len(df)}/{n_before} rows with NRMSE > {args.cap}")
            else:
                df["nrmse"] = df["nrmse"].clip(upper=args.cap)
                n_capped = (df["nrmse"] == args.cap).sum()
                print(f"Capped {n_capped} rows at NRMSE = {args.cap}")

        agg = df.groupby(["scaffold", "P", "species"])["nrmse"].agg(
            ["count", "mean", "median", "std"]
        ).reset_index()
        agg["sem"] = agg["std"] / np.sqrt(agg["count"])
        agg["q25"] = df.groupby(["scaffold", "P", "species"])["nrmse"].quantile(0.25).values
        agg["q75"] = df.groupby(["scaffold", "P", "species"])["nrmse"].quantile(0.75).values
    else:
        agg = df

    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = plt.get_cmap("tab10")
    species_list = sorted(agg["species"].unique())

    for i, sp in enumerate(species_list):
        sub = agg[agg["species"] == sp].sort_values("P")
        x = sub["P"].values.astype(float)
        y = sub[args.stat].values.astype(float)
        if args.error_bar == "iqr":
            e = np.vstack([y - sub["q25"].values, sub["q75"].values - y])
        else:
            e = sub[args.error_bar].values.astype(float)
        ax.errorbar(
            x, y, yerr=e,
            marker="o", linestyle="--", linewidth=1.8, markersize=5.5,
            color=cmap(i % 10), alpha=0.85, capsize=4,
            label=f"{sp}",
        )

    ax.set_xlabel("Number of mechanistic equations ($P$)", fontsize=14)
    ax.set_ylabel("NRMSE (lower is better)", fontsize=14)
    ax.set_xticks(sorted(agg["P"].unique()))
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12, framealpha=0.9)
    fig.tight_layout()

    out_path = Path(args.out) if args.out else Path(args.csv_path).with_suffix(f".{args.format}")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
