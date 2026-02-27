from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pick_final_product_idx(obs_names: Optional[list[str]], P: int) -> int:
    # Prefer common final-product names if present, else default to last species.
    if obs_names is not None:
        lowered = [str(n).strip().lower() for n in obs_names]
        for key in ["m", "product", "final", "output"]:
            if key in lowered:
                return int(lowered.index(key))
    return int(P - 1)


def _plot_y_clean_overlay(
    out_dir: Path,
    t_obs: np.ndarray,
    y0: np.ndarray,
    y_seq: np.ndarray,
    u_seq: Optional[np.ndarray],
    obs_names: Optional[list[str]],
    n_plot: int,
) -> None:
    """
    One clean plot per sample:
      - all observed species overlaid (transparent)
      - final product highlighted (thicker)
      - optional vertical markers at timesteps with any bolus (from u_seq)
    """
    N, K, P = y_seq.shape
    n_plot = min(int(n_plot), N)

    t_obs = np.asarray(t_obs, dtype=np.float32)
    tt = t_obs[1:]  # y_seq corresponds to t1..tK

    final_idx = _pick_final_product_idx(obs_names, P)

    for i in range(n_plot):
        fig, ax = plt.subplots(figsize=(11, 5.5))

        # Plot all species (thin/transparent)
        for p in range(P):
            name = obs_names[p] if (obs_names is not None and p < len(obs_names)) else f"y[{p}]"
            lw = 2.6 if p == final_idx else 1.6
            a = 1.0 if p == final_idx else 0.55
            ax.plot(tt, y_seq[i, :, p], linewidth=lw, alpha=a, label=name if p == final_idx else None)

        # Optional: mark bolus timesteps (any channel nonzero)
        if u_seq is not None:
            u_i = np.asarray(u_seq[i], dtype=np.float32)  # (K,U)
            bolus_bins = np.where(u_i.sum(axis=1) > 0)[0]  # indices in 0..K-1
            for k in bolus_bins:
                ax.axvline(tt[k], linewidth=1.0, alpha=0.15)

        # Baselines at y0 (optional, subtle)
        # (Plot only for final product to avoid clutter)
        ax.axhline(float(y0[i, final_idx]), linestyle="--", linewidth=1.2, alpha=0.6)

        final_name = (
            obs_names[final_idx]
            if (obs_names is not None and final_idx < len(obs_names))
            else f"y[{final_idx}]"
        )

        ax.set_title(f"Observed species (overlay) — sample {i} | highlighted: {final_name}")
        ax.set_xlabel("time")
        ax.set_ylabel("concentration")
        ax.grid(True, alpha=0.25)

        # Legend only for highlighted final product (keeps it clean)
        ax.legend(loc="best")

        fig.tight_layout()
        fig.savefig(out_dir / f"y_overlay_{i:03d}.png", dpi=170)
        plt.close(fig)


def _plot_u_total(
    out_dir: Path,
    t_obs: np.ndarray,
    u_seq: np.ndarray,
    n_plot: int,
) -> None:
    """
    One clean plot per sample:
      - total bolus per timestep: sum_u[k] = sum_j u_seq[k,j]
    """
    N, K, U = u_seq.shape
    n_plot = min(int(n_plot), N)

    t_obs = np.asarray(t_obs, dtype=np.float32)
    tt = t_obs[1:]  # bins correspond to intervals ending at t1..tK

    for i in range(n_plot):
        total_u = np.asarray(u_seq[i], dtype=np.float32).sum(axis=1)  # (K,)

        fig, ax = plt.subplots(figsize=(11, 3.8))

        # Bar is nice because "bolus bins" are discrete
        ax.bar(tt, total_u, width=(tt[1] - tt[0]) * 0.85 if K > 1 else 1.0, alpha=0.8)

        ax.set_title(f"Total bolus per timestep — sample {i}")
        ax.set_xlabel("time")
        ax.set_ylabel("Σ bolus (all channels)")
        ax.grid(True, axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(out_dir / f"u_total_{i:03d}.png", dpi=170)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=str, help="Path to .npz dataset")
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: <dataset>_preview/)",
    )
    ap.add_argument("--n-plot", type=int, default=3, help="How many samples to plot")
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    d = _load_npz(ds_path)

    _print_summary(d)

    y0 = np.asarray(d["y0"], dtype=np.float32)       # (N,P)
    u_seq = np.asarray(d["u_seq"], dtype=np.float32) # (N,K,U)
    y_seq = np.asarray(d["y_seq"], dtype=np.float32) # (N,K,P)
    t_obs = np.asarray(d["t_obs"], dtype=np.float32) # (K+1,)

    control_names = _as_str_array(d.get("control_names", None))
    obs_names = _as_str_array(d.get("obs_names", None))

    _nonzero_u_stats(u_seq, control_names=control_names)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else ds_path.with_suffix("").with_name(ds_path.stem + "_preview")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean overlay plot (all observed species + highlight final product + bolus markers)
    _plot_y_clean_overlay(
        out_dir,
        t_obs=t_obs,
        y0=y0,
        y_seq=y_seq,
        u_seq=u_seq,
        obs_names=obs_names,
        n_plot=args.n_plot,
    )

    # Clean total-bolus plot (sum over channels per timestep)
    _plot_u_total(
        out_dir,
        t_obs=t_obs,
        u_seq=u_seq,
        n_plot=args.n_plot,
    )

    print(f"\nSaved preview plots to: {out_dir}")


if __name__ == "__main__":
    main()