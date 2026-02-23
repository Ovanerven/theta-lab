import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.interpolate import interp1d

from src.sim.benchmark_models import FullModel
from src.sim.syndata_simulator_ODE import (
    generate_random_bolus_events,
    simulate_chain_with_bolus,
    single_event_generator,
)


def _parse_int_list(value: str) -> List[int]:
    value = value.strip()
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def generate_training_dataset(
    *,
    n_samples: int = 1000,
    t_span: float = 300.0,
    n_steps: int = 600,  # K = n_steps - 1
    control_indices: Optional[List[int]] = None,
    obs_indices: Optional[List[int]] = None,
    zero_init: bool = False,
    tail: float = 0.0,
    output_file: Optional[str] = None,
    seed: Optional[int] = 42,
    k_noise: float = 0.0,
) -> None:
    """
    Discrete-time dataset for sequence models (RNN / SSM / Transformer).

    Problem formulation:
      x(t) ∈ R^n: full chemical state (n_states_full)
      y(t) = H(x(t)) ∈ R^p: observations via index selection (p_obs << n)
      u(t) ∈ R^d: d-channel control inputs (boluses)
      θ: constant kinetics parameters (optionally perturbed per sample)

    Saved outputs (.npz):
      y0            : (N, p_obs)      observed state y(t0)
      u_seq         : (N, K, d_in)    binned bolus inputs per interval
      y_seq         : (N, K, p_obs)   observed states y(t1)..y(tK)
      t_obs        : (K+1,)          time grid [t0..tK]
      control_indices: (d_in,)        which species are controllable (indices into full state)
      obs_indices   : (p_obs,)        which species are observed (indices into full state)
      theta_true    : (n_params,)     base kinetics parameters
      theta_full    : (N, n_params)   only if k_noise > 0 (per-sample kinetics)
    """

    if output_file is None:
        raise ValueError("output_file must be provided")

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_states_full, n_params_full, names_full = FullModel(None, None, None, dim=True)

    if obs_indices is None:
        obs_indices = [0, 3, 6, 9, 12]  # default: reduced 5-state chain
    obs_indices = np.asarray(obs_indices, dtype=np.int64)
    p_obs = int(obs_indices.shape[0])

    if control_indices is None:
        control_indices = list(range(n_states_full))
    else:
        control_indices = list(control_indices)
    d_in = len(control_indices)

    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")

    t_obs = np.linspace(0.0, t_span, n_steps).astype(np.float32)
    K = n_steps - 1

    y0 = np.zeros((n_samples, p_obs), dtype=np.float32)
    u_seq = np.zeros((n_samples, K, d_in), dtype=np.float32)
    y_seq = np.zeros((n_samples, K, p_obs), dtype=np.float32)

    rng = np.random.default_rng(seed)

    allowed_names = [names_full[idx] for idx in control_indices]
    name_to_channel = {names_full[idx]: j for j, idx in enumerate(control_indices)}

    print(f"Generating {n_samples} samples | K={K}, p_obs={p_obs}, d_in={d_in}")

    theta_true = rng.uniform(0.5, 1.5, size=n_params_full).astype(np.float32)

    theta_full = None
    if k_noise > 0.0:
        print(f"  adding kinetics noise with std={k_noise}")
        theta_full = np.repeat(theta_true[None, :], n_samples, axis=0).astype(np.float32)

    for i in range(n_samples):
        if theta_full is not None:
            theta_full[i] += rng.normal(0.0, k_noise, size=n_params_full).astype(np.float32)
            k_for_sim = theta_full[i]
        else:
            k_for_sim = theta_true

        events = generate_random_bolus_events(
            n_bolus=int(rng.integers(2, 50)),
            t_start=0.0,
            t_end=max(0.0, t_span - tail),
            amount_range=(0.5, 3.0),
            species_names=allowed_names,
            rng=rng,
        )

        x0_full = np.zeros((n_states_full,), dtype=np.float32) if zero_init else np.ones((n_states_full,), dtype=np.float32)

        t_solver, x_solver = simulate_chain_with_bolus(
            FullModel,
            k=k_for_sim,
            y0=x0_full,
            t_start=0.0,
            t_end=t_span,
            dt=0.01,
            bolus_gen=single_event_generator(events),
            species_names=names_full,
        )

        x_solver = np.asarray(x_solver, dtype=np.float32)
        t_solver = np.asarray(t_solver, dtype=np.float32)

        x_grid = interp1d(t_solver, x_solver, axis=0, kind="linear", fill_value="extrapolate")(t_obs).astype(np.float32)

        y_grid = x_grid[:, obs_indices]
        y0[i] = y_grid[0]
        y_seq[i] = y_grid[1:]

        # Bin bolus events into intervals [t_k, t_{k+1})
        u_bins = np.zeros((K, d_in), dtype=np.float32)
        for (t_bolus, name, amt) in events:
            k = int(np.searchsorted(t_obs, np.float32(t_bolus), side="right")) - 1
            k = max(0, min(k, K - 1))
            u_bins[k, name_to_channel[name]] += np.float32(amt)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--t-span", type=float, default=300.0)
    parser.add_argument("--n-steps", type=int, default=600)
    parser.add_argument("--control-indices", type=str, default=None)
    parser.add_argument("--obs-indices", type=str, default=None)
    parser.add_argument("--tail", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--zero-init", action="store_true", help="Use zero initial conditions instead of ones")
    parser.add_argument("--k-noise", type=float, default=0.0, help="Stddev of kinetics noise per sample")
    args = parser.parse_args()

    init_tag = "zeros" if args.zero_init else "ones"
    suffix = (
        f"N{args.n_samples}_"
        f"T{int(args.t_span)}_"
        f"steps{args.n_steps}_"
        f"{init_tag}_"
        f"knoise{args.k_noise}"
    )

    if args.output_file is None:
        output_file = f"datasets/{suffix}.npz"
    else:
        p = Path(args.output_file)
        if p.suffix != ".npz":
            p = p.with_suffix(".npz")
        if not p.is_absolute() and p.parts and p.parts[0] != "datasets":
            p = Path("datasets") / p
        output_file = str(p)

    control_indices = _parse_int_list(args.control_indices) if args.control_indices else None
    obs_indices = _parse_int_list(args.obs_indices) if args.obs_indices else None

    generate_training_dataset(
        n_samples=args.n_samples,
        t_span=args.t_span,
        n_steps=args.n_steps,
        control_indices=control_indices,
        obs_indices=obs_indices,
        tail=args.tail,
        output_file=output_file,
        seed=args.seed,
        zero_init=args.zero_init,
        k_noise=args.k_noise,
    )


if __name__ == "__main__":
    main()
