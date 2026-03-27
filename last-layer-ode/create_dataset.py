import argparse
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import interp1d

from sim.benchmark_models import FullModel
from sim.MOF_model import MOF_Synthesis
from sim.syndata_simulator_ODE import simulate_chain_with_bolus, simulate_ivp_with_bolus, single_event_generator

SIM_MODELS = {
    "full13":        FullModel,
    "mof_synthesis": MOF_Synthesis,
}


# ============================================================
# Model-specific configurations
# ============================================================

@dataclass
class ModelConfig:
    """Per-model defaults for dataset generation."""
    # Parameter sampling: list of (lo, hi) per parameter index.
    # If None, falls back to global uniform(0.5, 1.5).
    param_ranges: Optional[List[Tuple[float, float]]] = None

    # Full-state initial conditions (length = n_states_full).
    # If None, uses zero_init / ones logic as before.
    x0_default: Optional[List[float]] = None

    # Per-control-channel bolus amount ranges: dict mapping
    # species name -> (lo, hi).  Channels not listed use the global default.
    bolus_ranges: Optional[Dict[str, Tuple[float, float]]] = None

    # Global bolus amount range fallback
    bolus_default: Tuple[float, float] = (0.5, 3.0)

    # Max number of boluses per trajectory
    bolus_count_range: Tuple[int, int] = (2, 50)

    # Simulator backend: "rk4" (fixed-step) or "ivp" (adaptive BDF via solve_ivp)
    simulator: str = "rk4"


def _mof_max_config() -> ModelConfig:
    """
    Physically informed parameter ranges for MOF_Mechanistic_Max (44 params, 18 states).

    Parameter order (from MOF_model.py):
      0  ka_L       acid dissoc linker
      1  kma_L      reprotonat linker
      2  ka_B       acid dissoc modulator
      3  kma_B      reprotonat modulator
      4  k1f        M+Lm -> C1
      5  k1r        C1 -> M+Lm
      6  k2f        C1+Lm -> C2
      7  k2r        C2 reverse
      8  k3f        C2+Lm -> C3
      9  k3r        C3 reverse
      10 kBf        M+Bm -> CB
      11 kBr        CB reverse
      12 nu         exponent C2->P
      13 kp_f       cluster formation
      14 kp_r       cluster dissolution
      15 m          condensation exponent
      16 kcond      condensation rate
      17 alphaA     autocatalytic factor
      18 nA         Hill exponent (cryst)
      19 KA         Hill half-sat (cryst)
      20 kcryst     crystallization rate
      21 Peq0       equilibrium baseline
      22 Eeq        activation energy
      23 betaH      proton modifier
      24 betaB      modulator modifier
      25 betaS      solvent modifier
      26 Tmin       temp clamp lower
      27 Tmax       temp clamp upper
      28 kJ         nucleation prefactor
      29 Bcnt       CNT barrier
      30 pJ         nucleation exponent
      31 ksec       secondary nucleation
      32 qsec       secondary nucleation exp
      33 shape      shape factor
      34 kagg       aggregation
      35 kg         growth rate
      36 gG         growth exponent
      37 KiB        modulator inhibition
      38 kdiss      dissolution
      39 chi        consumption factor
      40 kDf        defect formation
      41 KD         defect Hill half-sat
      42 kDa        defect annealing
      43 EaD        defect annealing Ea
    """
    param_ranges = [
        (0.1, 2.0),    # 0  ka_L
        (0.1, 2.0),    # 1  kma_L
        (0.1, 2.0),    # 2  ka_B
        (0.1, 2.0),    # 3  kma_B
        (0.05, 0.8),   # 4  k1f
        (0.01, 0.5),   # 5  k1r
        (0.05, 0.8),   # 6  k2f
        (0.01, 0.5),   # 7  k2r
        (0.02, 0.5),   # 8  k3f
        (0.01, 0.3),   # 9  k3r
        (0.02, 0.5),   # 10 kBf
        (0.01, 0.3),   # 11 kBr
        (1.5, 2.5),    # 12 nu       — should be ~2 (dimerization-like)
        (0.005, 0.2),   # 13 kp_f
        (0.01, 0.3),   # 14 kp_r
        (1.0, 2.0),    # 15 m        — condensation order ~1-2
        (0.01, 0.3),   # 16 kcond
        (0.1, 2.0),    # 17 alphaA   — autocatalytic strength
        (1.0, 3.0),    # 18 nA       — Hill exponent
        (0.5, 5.0),    # 19 KA       — Hill half-sat
        (0.05, 0.5),   # 20 kcryst
        (0.005, 0.05), # 21 Peq0     — equilibrium baseline
        (0.5, 3.0),    # 22 Eeq      — careful: enters exp(-Eeq/T), with _R_GAS=1 and T~350
        (0.0, 0.5),    # 23 betaH
        (0.0, 0.5),    # 24 betaB
        (0.0, 0.5),    # 25 betaS
        (280.0, 300.0), # 26 Tmin    — CRITICAL: must be physical temperature
        (400.0, 500.0), # 27 Tmax    — CRITICAL
        (0.01, 0.5),   # 28 kJ       — nucleation prefactor (keep moderate)
        (0.5, 5.0),    # 29 Bcnt     — CNT barrier (very sensitive, keep modest)
        (0.0, 1.5),    # 30 pJ       — nucleation exponent on (S-1)
        (0.01, 0.3),   # 31 ksec     — secondary nucleation
        (1.0, 2.0),    # 32 qsec     — secondary nucleation exponent
        (0.8, 1.2),    # 33 shape    — geometric factor, near 1
        (0.0001, 0.005), # 34 kagg     — aggregation (keep small)
        (0.01, 0.3),   # 35 kg       — growth rate
        (1.0, 2.0),    # 36 gG       — growth exponent
        (0.1, 2.0),    # 37 KiB      — inhibition constant
        (0.0001, 0.005), # 38 kdiss    — dissolution (smaller than growth)
        (0.001, 0.03), # 39 chi      — consumption factor (sensitive with SA)
        (0.01, 0.3),   # 40 kDf      — defect formation
        (0.1, 1.0),    # 41 KD       — defect Hill half-sat
        (0.01, 0.3),   # 42 kDa      — defect annealing
        (0.5, 3.0),    # 43 EaD      — defect Ea
    ]

    # Initial conditions for all 18 states:
    # [M, LH, Lm, H, BH, Bm, C1, C2, C3, CB, P, A, N, R, D, T, Solv, I]
    x0_default = [
        1.0,   # M   — start with some metal
        1.0,   # LH  — start with some protonated linker
        0.1,   # Lm  — small amount deprotonated
        0.01,  # H   — low initial proton concentration
        0.5,   # BH  — some modulator
        0.05,  # Bm  — small amount deprotonated modulator
        0.0,   # C1  — no complexes initially
        0.0,   # C2
        0.0,   # C3
        0.0,   # CB
        0.0,   # P   — no clusters
        0.0,   # A   — no amorphous
        0.0,   # N   — no crystals
        0.0,   # R   — no size
        0.0,   # D   — no defects
        350.0, # T   — ~350K (typical solvothermal)
        1.0,   # Solv — solvent quality = 1
        0.0,   # I   — no impurity
    ]

    # Bolus amounts per channel — reagents get normal amounts,
    # temperature gets small perturbations (±20K-ish)
    bolus_ranges = {
        "M":    (0.2, 2.0),
        "LH":   (0.2, 2.0),
        "BH":   (0.1, 1.5),
        "T":    (5.0, 30.0),    # temperature step changes in K
        "Solv": (0.1, 1.0),
        "I":    (0.05, 0.5),
    }

    return ModelConfig(
        param_ranges=param_ranges,
        x0_default=x0_default,
        bolus_ranges=bolus_ranges,
        bolus_default=(0.2, 2.0),
        bolus_count_range=(3, 30),  # fewer boluses for stability
    )


def _mof_synthesis_config() -> ModelConfig:
    """
    Fixed kinetic parameters from supervisor's MOF_synthesis.py script.

    Parameters in order (16):
      k_deprot=5.0, k_prot=1.0, k_oli=3.0, k_cap=2.0, k_uncap=0.5,
      K_I=0.1, knuc_A=10.0, kgro_A=1.0, kagg_A=1.0, n_A=3.0,
      knuc_C=0.5, kgro_C=4.0, kagg_C=1.0, n_C=1.5, a=1.0, b=1.0

    Control channels: Base (idx 4) and Mod (idx 5) — matching the
    supervisor's dosing strategy.  Use --t-span 30 when generating.
    """
    supervisor_params = [5.0, 1.0, 3.0, 2.0, 0.5, 0.1, 10.0, 1.0, 1.0, 3.0,
                         0.5, 4.0, 1.0, 1.5, 1.0, 1.0]
    param_ranges = [(v, v) for v in supervisor_params]

    # [Met, LigH, Lig_minus, H_plus, Base, Mod, SBU, SBU_capped, Nuc_A, Am, Nuc_C, MOF_C]
    x0_default = [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    bolus_ranges = {
        "Base": (0.1, 2.0),
        "Mod":  (0.1, 1.0),
    }

    return ModelConfig(
        param_ranges=param_ranges,
        x0_default=x0_default,
        bolus_ranges=bolus_ranges,
        bolus_default=(0.1, 1.0),
        bolus_count_range=(2, 8),
        simulator="ivp",
    )


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "full13":        ModelConfig(),
    "mof_synthesis": _mof_synthesis_config(),
}


# ============================================================
# Utility functions (unchanged)
# ============================================================

def _parse_int_list(value: str) -> List[int]:
    value = value.strip()
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _generate_random_events_idx(
    *,
    rng: np.random.Generator,
    n_bolus: int,
    t_start: float,
    t_end: float,
    amount_range: Tuple[float, float],
    n_channels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random bolus schedule in integer-channel form.

    Returns (t_event, ch_event, amt_event), each shape (n_bolus,).
    """
    t_event = rng.uniform(t_start, t_end, size=n_bolus).astype(np.float32)
    t_event.sort()
    ch_event = rng.integers(0, n_channels, size=n_bolus, dtype=np.int64)
    amt_event = rng.uniform(amount_range[0], amount_range[1], size=n_bolus).astype(np.float32)
    return t_event, ch_event, amt_event


def _generate_random_events_per_channel(
    *,
    rng: np.random.Generator,
    n_bolus: int,
    t_start: float,
    t_end: float,
    n_channels: int,
    control_names: np.ndarray,
    bolus_ranges: Optional[Dict[str, Tuple[float, float]]],
    bolus_default: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random bolus schedule with per-channel amount ranges.

    Returns (t_event, ch_event, amt_event), each shape (n_bolus,).
    """
    t_event = rng.uniform(t_start, t_end, size=n_bolus).astype(np.float32)
    t_event.sort()
    ch_event = rng.integers(0, n_channels, size=n_bolus, dtype=np.int64)

    amt_event = np.empty(n_bolus, dtype=np.float32)
    for j in range(n_bolus):
        ch = int(ch_event[j])
        name = str(control_names[ch])
        lo, hi = bolus_default
        if bolus_ranges is not None and name in bolus_ranges:
            lo, hi = bolus_ranges[name]
        amt_event[j] = rng.uniform(lo, hi)

    return t_event, ch_event, amt_event


def _bin_events_to_u_seq(
    *,
    t_obs: np.ndarray,  # (K+1,)
    t_event: np.ndarray,
    ch_event: np.ndarray,
    amt_event: np.ndarray,
    d_in: int,
) -> np.ndarray:
    """Bin events into intervals [t_k, t_{k+1}) as u_seq[k, ch] += amt."""
    K = int(t_obs.shape[0] - 1)
    u_bins = np.zeros((K, d_in), dtype=np.float32)
    for t_bolus, ch, amt in zip(t_event, ch_event, amt_event):
        k = int(np.searchsorted(t_obs, np.float32(t_bolus), side="right")) - 1
        k = max(0, min(k, K - 1))
        u_bins[k, int(ch)] += np.float32(amt)
    return u_bins


def _sample_params(
    rng: np.random.Generator,
    n_params: int,
    param_ranges: Optional[List[Tuple[float, float]]],
) -> np.ndarray:
    """Sample parameters from model-specific ranges or fallback to U(0.5, 1.5)."""
    if param_ranges is not None:
        assert len(param_ranges) == n_params, (
            f"param_ranges has {len(param_ranges)} entries but model has {n_params} params"
        )
        theta = np.empty(n_params, dtype=np.float32)
        for i, (lo, hi) in enumerate(param_ranges):
            theta[i] = rng.uniform(lo, hi)
        return theta
    else:
        return rng.uniform(0.5, 1.5, size=n_params).astype(np.float32)


def _build_x0(
    n_states_full: int,
    zero_init: bool,
    x0_default: Optional[List[float]],
) -> np.ndarray:
    """Build initial condition vector, using model-specific defaults if available."""
    if x0_default is not None:
        assert len(x0_default) == n_states_full, (
            f"x0_default has {len(x0_default)} entries but model has {n_states_full} states"
        )
        return np.asarray(x0_default, dtype=np.float32)
    elif zero_init:
        return np.zeros(n_states_full, dtype=np.float32)
    else:
        return np.ones(n_states_full, dtype=np.float32)


# ============================================================
# Main dataset generation
# ============================================================

def generate_training_dataset(
    *,
    model_fn=None,
    model_name: str = "full13",
    n_samples: int = 1000,
    t_span: float = 300.0,
    n_steps: int = 600,
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

    Now supports model-specific parameter ranges, initial conditions,
    and per-channel bolus amount ranges via MODEL_CONFIGS.
    """

    if output_file is None:
        raise ValueError("output_file must be provided")

    if model_fn is None:
        model_fn = FullModel

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_states_full, n_params_full, names_full = model_fn(None, None, None, dim=True)
    names_full = list(names_full)

    # Load model-specific config
    mcfg = MODEL_CONFIGS.get(model_name, ModelConfig())

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

    control_indices = np.asarray(control_indices, dtype=np.int64)
    control_names = np.asarray([names_full[int(idx)] for idx in control_indices], dtype="<U16")
    obs_names = np.asarray([names_full[int(idx)] for idx in obs_indices], dtype="<U16")

    print(f"Generating {n_samples} samples | K={K}, p_obs={p_obs}, d_in={d_in}")
    print(f"Model config: {model_name} | param_ranges={'custom' if mcfg.param_ranges else 'default'}")
    print(f"Control channels: {list(control_names)}")
    print(f"Observed species: {list(obs_names)}")

    # Sample parameters from model-specific ranges
    theta_true = _sample_params(rng, n_params_full, mcfg.param_ranges)

    theta_full = None
    if k_noise > 0.0:
        print(f"  adding kinetics noise with std={k_noise}")
        theta_full = np.repeat(theta_true[None, :], n_samples, axis=0).astype(np.float32)

    # Build initial conditions
    x0_template = _build_x0(n_states_full, zero_init, mcfg.x0_default)

    try:
        _tqdm_mod = importlib.import_module("tqdm.auto")
        sample_iter = _tqdm_mod.tqdm(range(n_samples), desc="Simulating samples", unit="sample")
    except Exception:
        sample_iter = range(n_samples)

    n_failed = 0
    for i in sample_iter:
        if theta_full is not None:
            theta_full[i] += rng.normal(0.0, k_noise, size=n_params_full).astype(np.float32)
            k_for_sim = theta_full[i]
        else:
            k_for_sim = theta_true

        bolus_lo, bolus_hi = mcfg.bolus_count_range
        n_bolus = int(rng.integers(bolus_lo, bolus_hi))

        # Use per-channel bolus ranges if available
        if mcfg.bolus_ranges is not None:
            t_event, ch_event, amt_event = _generate_random_events_per_channel(
                rng=rng,
                n_bolus=n_bolus,
                t_start=0.0,
                t_end=max(0.0, t_span - tail),
                n_channels=d_in,
                control_names=control_names,
                bolus_ranges=mcfg.bolus_ranges,
                bolus_default=mcfg.bolus_default,
            )
        else:
            t_event, ch_event, amt_event = _generate_random_events_idx(
                rng=rng,
                n_bolus=n_bolus,
                t_start=0.0,
                t_end=max(0.0, t_span - tail),
                amount_range=mcfg.bolus_default,
                n_channels=d_in,
            )

        # simulator consumes (time, species_name, amount)
        events = [(float(t), str(control_names[int(ch)]), float(a)) for t, ch, a in zip(t_event, ch_event, amt_event)]

        x0_full = x0_template.copy()

        if mcfg.simulator == "ivp":
            t_solver, x_solver = simulate_ivp_with_bolus(
                model_fn,
                k=k_for_sim,
                y0=x0_full,
                t_start=0.0,
                t_end=t_span,
                bolus_gen=single_event_generator(events),
                species_names=names_full,
            )
        else:
            t_solver, x_solver = simulate_chain_with_bolus(
                model_fn,
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

        # Check for NaN / massive negative values (solver instability)
        if np.any(np.isnan(x_solver)) or np.any(x_solver < -100):
            n_failed += 1
            # Fill with zeros so we can still save; you may want to resample instead
            y0[i] = 0.0
            y_seq[i] = 0.0
            u_seq[i] = 0.0
            continue

        x_grid = interp1d(t_solver, x_solver, axis=0, kind="linear", fill_value="extrapolate")(t_obs).astype(np.float32)

        y_grid = x_grid[:, obs_indices]
        y0[i] = y_grid[0]
        y_seq[i] = y_grid[1:]

        u_seq[i] = _bin_events_to_u_seq(
            t_obs=t_obs,
            t_event=t_event,
            ch_event=ch_event,
            amt_event=amt_event,
            d_in=d_in,
        )

    if n_failed > 0:
        print(f"WARNING: {n_failed}/{n_samples} samples had NaN or large negatives (zeroed out)")

    save_kwargs = dict(
        y0=y0,
        u_seq=u_seq,
        y_seq=y_seq,
        t_obs=t_obs,
        control_indices=control_indices,
        obs_indices=obs_indices,
        names_full=np.asarray(names_full, dtype="<U16"),
        control_names=control_names,
        obs_names=obs_names,
        n_states_full=np.int64(n_states_full),
        n_params_full=np.int64(n_params_full),
        theta_true=theta_true,
    )
    if theta_full is not None:
        save_kwargs["theta_full"] = theta_full

    np.savez(str(out_path), **save_kwargs)

    print(f"Saved dataset to {out_path}")
    print(f"y0:{y0.shape}, u_seq:{u_seq.shape}, y_seq:{y_seq.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="full13",
                        choices=list(SIM_MODELS.keys()),
                        help="Simulation model to use for data generation. "
                             "Use --t-span 30 for mof_synthesis.")
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
        model_fn=SIM_MODELS[args.model],
        model_name=args.model,
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