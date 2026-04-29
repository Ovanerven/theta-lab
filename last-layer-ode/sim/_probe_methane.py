"""Probe trajectory script: run GRI30/Kazakov/Smooke from a stoichiometric
methane/air mix and compare long-horizon plateaus.

Run from project root:
    python last-layer-ode/sim/_probe_methane.py --t-end 50 --out results/methane_probe/overlay.png
"""
import sys
from pathlib import Path
import argparse
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.explicit_methane_models import (
    GRI30_FullModel, Kazakov_MiddleModel, Smooke_ReducedModel,
)


def _build_ic(names: list[str]) -> np.ndarray:
    y0 = np.zeros(len(names), dtype=float)
    name_to_idx = {n: i for i, n in enumerate(names)}
    y0[name_to_idx["CH4"]] = 1.0
    y0[name_to_idx["O2"]] = 2.0
    y0[name_to_idx["N2"]] = 7.52
    return y0


def _simulate(model, label: str, t_end: float, n_points: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n_states, n_params, names, _observed, *_ = model(None, None, None, dim=True)
    print(f"\n=== {label}: {n_states} states, {n_params} params ===")

    y0 = _build_ic(list(names))
    name_to_idx = {n: i for i, n in enumerate(names)}

    # All-ones effective k
    k = np.ones(n_params, dtype=float)

    sol = solve_ivp(
        lambda t, y: model(t, y, k),
        [0.0, t_end], y0, method="BDF", rtol=1e-6, atol=1e-9,
        t_eval=np.linspace(0, t_end, n_points),
    )
    if not sol.success:
        raise RuntimeError(f"{label} solver failed: {sol.message}")

    species = ["CH4", "O2", "CO", "CO2", "H2O", "OH"]
    series = {}
    for sp in species:
        if sp in name_to_idx:
            series[sp] = sol.y[name_to_idx[sp], :]
    return sol.t, series


def plot_overlay(results: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]], out_path: Path) -> None:
    species = ["CH4", "O2", "CO", "CO2", "H2O", "OH"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    axes = axes.reshape(-1)

    for ax, sp in zip(axes, species):
        for label, (t, series) in results.items():
            if sp not in series:
                continue
            ax.plot(t, series[sp], label=label, linewidth=2)
        ax.set_title(sp)
        ax.grid(True, alpha=0.25)

    axes[-1].legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Time")
    fig.suptitle("Methane autoignition: model comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-end", type=float, default=50.0)
    parser.add_argument("--n-points", type=int, default=400)
    parser.add_argument("--out", type=str, default="results/methane_probe/overlay.png")
    args = parser.parse_args()

    results = {}
    for label, model in (
        ("GRI30", GRI30_FullModel),
        ("Kazakov", Kazakov_MiddleModel),
        ("Smooke", Smooke_ReducedModel),
    ):
        t, series = _simulate(model, label, t_end=args.t_end, n_points=args.n_points)
        results[label] = (t, series)

    plot_overlay(results, Path(args.out))
    print(f"Saved overlay plot to {args.out}")
