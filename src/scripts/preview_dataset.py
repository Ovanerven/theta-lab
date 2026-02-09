from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.sim.benchmark_models import FullModel


def _plot_sample(
    *,
    out_path: Path,
    x0: np.ndarray,         # (P,)
    u_seq: np.ndarray,      # (K,U)
    t_obs: np.ndarray,      # (K+1,) time grid
    y_seq: np.ndarray,      # (K,P)
    title: str,
    channel_names: List[str] | None = None,
    input_names: List[str] | None = None,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    K, P = y_seq.shape
    U = u_seq.shape[1]

    t = t_obs  # (K+1,) already provided

    # y including t0
    y_full = np.vstack([x0[None, :], y_seq])  # (K+1,P)

    if channel_names is None:
        channel_names = [f"y{i}" for i in range(P)]
    if input_names is None:
        input_names = [f"u{i}" for i in range(U)]

    # Layout: 1 row for inputs + P rows for outputs
    fig, axes = plt.subplots(1 + P, 1, figsize=(12, 2.0 * (1 + P)), sharex=True)

    # Inputs: plot binned impulses at t1..tK
    ax_u = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, U))
    for u in range(U):
        # sparse impulses -> stem-like
        nonzero = np.nonzero(u_seq[:, u] != 0.0)[0]
        if nonzero.size > 0:
            ax_u.vlines(t[nonzero + 1], 0.0, u_seq[nonzero, u], linewidth=2, label=input_names[u], color=colors[u])
    ax_u.set_ylabel("u")
    ax_u.grid(True, alpha=0.25)
    if U > 0:
        ax_u.legend(loc="upper right")

    # Outputs
    for p in range(P):
        ax = axes[1 + p]
        is_last = (p == P - 1)
        ax.plot(t, y_full[:, p], linewidth=2, linestyle="--" if is_last else "-", label=channel_names[p])
        ax.set_ylabel(channel_names[p])
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("time")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Load an existing dataset and visualize sample trajectories.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to .npz dataset file")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of samples to visualize (default: 5)")
    parser.add_argument("--out-dir", type=str, default="preview", help="Output directory for plots")
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    d = np.load(str(dataset_path), allow_pickle=False)
    y0 = d["y0"]           # (N,P)
    u_seq = d["u_seq"]     # (N,K,U)
    y_seq = d["y_seq"]     # (N,K,P)
    t_obs = d["t_obs"]     # (K+1,) time grid
    control_indices = d["control_indices"]  # (U,)

    N = y0.shape[0]
    n_to_plot = min(args.n_samples, N)

    # Get species names from FullModel
    _, _, species_names = FullModel(None, None, None, dim=True)

    # Map control indices to names
    input_names = [species_names[idx] for idx in control_indices]

    # For this benchmark, observables are always [A,D,G,J,M]
    output_names = ["A", "D", "G", "J", "M"]

    print(f"Loaded dataset: {dataset_path}")
    print(f"  Total samples: {N}")
    print(f"  Visualizing: {n_to_plot} samples")
    print(f"  Output directory: {out_dir}")

    for i in range(n_to_plot):
        _plot_sample(
            out_path=out_dir / f"sample_{i:03d}.png",
            x0=y0[i],
            u_seq=u_seq[i],
            t_obs=t_obs,
            y_seq=y_seq[i],
            title=f"Sample {i} from {dataset_path.name}",
            channel_names=output_names,
            input_names=input_names,
        )

    print(f"Wrote {n_to_plot} preview plots to {out_dir}")


if __name__ == "__main__":
    main()
