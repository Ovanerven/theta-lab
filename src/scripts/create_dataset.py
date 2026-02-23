from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.interpolate import interp1d

from src.sim.benchmark_models import FullModel
from src.sim.syndata_simulator_ODE import simulate_chain_with_bolus, single_event_generator


def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def default_outfile(*, n_samples: int, t_span: float, n_steps: int, zero_init: bool, k_noise: float) -> str:
    init_tag = "zeros" if zero_init else "ones"
    name = f"N{n_samples}_T{int(t_span)}_steps{n_steps}_{init_tag}_knoise{k_noise}.npz"
    return str(Path("datasets") / name)


def normalize_outfile(path_str: str) -> Path:
    p = Path(path_str)
    if p.suffix != ".npz":
        p = p.with_suffix(".npz")
    if not p.is_absolute() and (not p.parts or p.parts[0] != "datasets"):
        p = Path("datasets") / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def generate_training_dataset(
    *,
    n_samples: int = 1000,
    t_span: float = 300.0,
    n_steps: int = 600,  # K = n_steps - 1
    control_indices: Optional[List[int]] = None,
    zero_init: bool = False,
    tail: float = 0.0,  # kept for API compatibility; unused in grid-sampled boluses
    output_file: str,
    seed: int = 42,
    k_noise: float = 0.0,
) -> None:
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")

    out_path = normalize_outfile(output_file)

    n_state, n_param, names = FullModel(None, None, None, dim=True)
    obs_indices = np.arange(n_state, dtype=np.int64)  # observe all species in file
    P = int(obs_indices.size)

    control_indices = list(range(n_state)) if control_indices is None else list(control_indices)
    U = len(control_indices)
    control_names = [names[i] for i in control_indices]

    t_obs = np.linspace(0.0, t_span, n_steps).astype(np.float32)
    K = n_steps - 1

    y0 = np.zeros((n_samples, P), dtype=np.float32)
    y_seq = np.zeros((n_samples, K, P), dtype=np.float32)
    u_seq = np.zeros((n_samples, K, U), dtype=np.float32)

    rng = np.random.default_rng(seed)
    theta_true = rng.uniform(0.5, 1.5, size=n_param).astype(np.float32)

    theta_full = None
    if k_noise > 0.0:
        theta_full = np.repeat(theta_true[None, :], n_samples, axis=0).astype(np.float32)

    x0_full = np.zeros(n_state, dtype=np.float32) if zero_init else np.ones(n_state, dtype=np.float32)

    print(f"Generating {n_samples} samples | K={K}, P={P}, U={U} | out={out_path}")

    for i in range(n_samples):
        k_for_sim = theta_true
        if theta_full is not None:
            theta_full[i] = theta_true + rng.normal(0.0, k_noise, size=n_param).astype(np.float32)
            k_for_sim = theta_full[i]

        # Sample boluses directly on the observation grid => exact alignment with model jump-times.
        n_bolus = int(rng.integers(2, 50))
        k_events = rng.integers(0, K, size=n_bolus)
        u_ch = rng.integers(0, U, size=n_bolus)
        amt = rng.uniform(0.5, 3.0, size=n_bolus).astype(np.float32)

        u_bins = np.zeros((K, U), dtype=np.float32)
        for k_e, ch, a in zip(k_events, u_ch, amt):
            u_bins[int(k_e), int(ch)] += float(a)

        # Convert u_bins -> simulator events (merged by construction, sorted by time)
        events = [
            (float(t_obs[k]), control_names[ch], float(u_bins[k, ch]))
            for k in range(K)
            for ch in range(U)
            if u_bins[k, ch] != 0.0
        ]

        t_solver, x_solver = simulate_chain_with_bolus(
            FullModel,
            k=k_for_sim,
            y0=x0_full,
            t_start=0.0,
            t_end=t_span,
            dt=0.01,
            bolus_gen=single_event_generator(events),
            species_names=names,
        )

        x_grid = interp1d(
            np.asarray(t_solver, dtype=np.float32),
            np.asarray(x_solver, dtype=np.float32),
            axis=0,
            kind="linear",
            fill_value="extrapolate",
        )(t_obs).astype(np.float32)

        y0[i] = x_grid[0, obs_indices]
        y_seq[i] = x_grid[1:, obs_indices]
        u_seq[i] = u_bins

        if (i + 1) % 100 == 0:
            print(f"  simulated {i+1}/{n_samples}")

    save_kwargs = dict(
        y0=y0,
        u_seq=u_seq,
        y_seq=y_seq,
        t_obs=t_obs,
        control_indices=np.asarray(control_indices, dtype=np.int64),
        obs_indices=obs_indices,
        theta_true=theta_true,
    )
    if theta_full is not None:
        save_kwargs["theta_full"] = theta_full

    np.savez(str(out_path), **save_kwargs)
    print(f"Saved dataset to {out_path}")
    print(f"y0:{y0.shape}, u_seq:{u_seq.shape}, y_seq:{y_seq.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--t-span", type=float, default=300.0)
    ap.add_argument("--n-steps", type=int, default=600)
    ap.add_argument("--control-indices", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-file", type=str, default=None)
    ap.add_argument("--zero-init", action="store_true")
    ap.add_argument("--k-noise", type=float, default=0.0)
    ap.add_argument("--tail", type=float, default=0.0)  # kept for compatibility
    args = ap.parse_args()

    output_file = args.output_file or default_outfile(
        n_samples=args.n_samples,
        t_span=args.t_span,
        n_steps=args.n_steps,
        zero_init=args.zero_init,
        k_noise=args.k_noise,
    )

    generate_training_dataset(
        n_samples=args.n_samples,
        t_span=args.t_span,
        n_steps=args.n_steps,
        control_indices=parse_int_list(args.control_indices),
        zero_init=args.zero_init,
        tail=args.tail,
        output_file=output_file,
        seed=args.seed,
        k_noise=args.k_noise,
    )


if __name__ == "__main__":
    main()
