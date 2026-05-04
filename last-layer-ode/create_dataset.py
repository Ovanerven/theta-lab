import argparse
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import interp1d
import cantera as ct
import torch

from sim.benchmark_models import FullModel
from sim.MOF_model import MOF_Synthesis
from sim.Single_enzyme import SingleEnzyme
from sim.explicit_methane_models import GRI30_FullModel, Kazakov_MiddleModel, Smooke_ReducedModel, Aramco_FullModel
from sim.syndata_simulator_ODE import simulate_chain_with_bolus, simulate_ivp_with_bolus, single_event_generator, simulate_cantera_with_bolus

# Lazy singleton — avoids instantiating the scaffold at import time.
_glycolysis_oracle22_scaffold = None
_glycolysis_reduced12_scaffold = None
_glycolysis_reduced8_scaffold = None
_glycolysis_reduced4_scaffold = None

def _get_glycolysis_oracle():
    global _glycolysis_oracle22_scaffold
    if _glycolysis_oracle22_scaffold is None:
        from sim.glycolysis import GlycolysisOracle22Scaffold
        _glycolysis_oracle22_scaffold = GlycolysisOracle22Scaffold()
    return _glycolysis_oracle22_scaffold


def _get_glycolysis_reduced12():
    global _glycolysis_reduced12_scaffold
    if _glycolysis_reduced12_scaffold is None:
        from sim.glycolysis import GlycolysisReduced12Scaffold
        _glycolysis_reduced12_scaffold = GlycolysisReduced12Scaffold()
    return _glycolysis_reduced12_scaffold


def _get_glycolysis_reduced8():
    global _glycolysis_reduced8_scaffold
    if _glycolysis_reduced8_scaffold is None:
        from sim.glycolysis import GlycolysisReduced8Scaffold
        _glycolysis_reduced8_scaffold = GlycolysisReduced8Scaffold()
    return _glycolysis_reduced8_scaffold


def _get_glycolysis_reduced4():
    global _glycolysis_reduced4_scaffold
    if _glycolysis_reduced4_scaffold is None:
        from sim.glycolysis import GlycolysisReduced4Scaffold
        _glycolysis_reduced4_scaffold = GlycolysisReduced4Scaffold()
    return _glycolysis_reduced4_scaffold


def GlycolysisOracle22(t: float, y: np.ndarray, k: np.ndarray, dim: bool = False):
    """
    22-state inhibited glycolysis oracle (scipy-compatible interface).

    States (22): Glc, G6P, F6P, FBP, GAP, DHAP, BPG13, PG3, PG2, PEP, Pyr, Lac,
                 ATP, ADP, NAD, NADH, I_HK, I_PFK, I_GAPDH, I_PGK, I_ENO, I_PK
    Parameters (33): k_hk … K_I_j, k_I_j_decay (see GlycolysisOracle22Scaffold)

    Observed species: Glc (0), Pyr (10), ATP (12), NADH (15)
    Recommended: --obs-indices 0,10,12,15 --control-indices 0,16,17,18,19,20,21
                 --t-span 20 --n-steps 400
    """
    if dim:
        names = [
            "Glc", "G6P", "F6P", "FBP", "GAP", "DHAP",
            "BPG13", "PG3", "PG2", "PEP", "Pyr", "Lac",
            "ATP", "ADP", "NAD", "NADH",
            "I_HK", "I_PFK", "I_GAPDH", "I_PGK", "I_ENO", "I_PK",
        ]
        return 22, 33, names
    scaffold = _get_glycolysis_oracle()
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0)
    k_t = torch.from_numpy(np.asarray(k, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        dy = scaffold(y_t, k_t).squeeze(0).numpy()
    return dy.astype(np.float64)


def GlycolysisReduced12(t: float, y: np.ndarray, k: np.ndarray, dim: bool = False):
    """12-state glycolysis reduced simulator (scipy-compatible interface)."""
    scaffold = _get_glycolysis_reduced12()
    if dim:
        return scaffold.P, scaffold.theta_dim, list(scaffold.state_names)
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0)
    k_t = torch.from_numpy(np.asarray(k, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        dy = scaffold(y_t, k_t).squeeze(0).numpy()
    return dy.astype(np.float64)


def GlycolysisReduced8(t: float, y: np.ndarray, k: np.ndarray, dim: bool = False):
    """8-state glycolysis reduced simulator (scipy-compatible interface)."""
    scaffold = _get_glycolysis_reduced8()
    if dim:
        return scaffold.P, scaffold.theta_dim, list(scaffold.state_names)
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0)
    k_t = torch.from_numpy(np.asarray(k, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        dy = scaffold(y_t, k_t).squeeze(0).numpy()
    return dy.astype(np.float64)


def GlycolysisReduced4(t: float, y: np.ndarray, k: np.ndarray, dim: bool = False):
    """4-state glycolysis reduced simulator (scipy-compatible interface)."""
    scaffold = _get_glycolysis_reduced4()
    if dim:
        return scaffold.P, scaffold.theta_dim, list(scaffold.state_names)
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0)
    k_t = torch.from_numpy(np.asarray(k, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        dy = scaffold(y_t, k_t).squeeze(0).numpy()
    return dy.astype(np.float64)


SIM_MODELS = {
    "full13":               FullModel,
    "mof_synthesis":        MOF_Synthesis,
    "single_enzyme":        SingleEnzyme,
    "gri30_full":           GRI30_FullModel,
    "kazakov_middle":       Kazakov_MiddleModel,
    "smooke_reduced_model": Smooke_ReducedModel,
    "aramco_30":            Aramco_FullModel,   # Natively routes to your explicit Python equations
    "glycolysis_oracle22":  GlycolysisOracle22,
    "glycolysis_reduced12": GlycolysisReduced12,
    "glycolysis_reduced8":  GlycolysisReduced8,
    "glycolysis_reduced4":  GlycolysisReduced4,
}

# ============================================================
# Model-specific configurations
# ============================================================

@dataclass
class ModelConfig:
    """Per-model defaults for dataset generation."""
    param_ranges: Optional[List[Tuple[float, float]]] = None
    x0_default: Optional[List[float]] = None
    bolus_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    bolus_default: Tuple[float, float] = (0.5, 3.0)
    bolus_count_range: Tuple[int, int] = (2, 50)
    simulator: str = "rk4"
    cantera_T: float = 1000.0
    cantera_T_range: Optional[Tuple[float, float]] = None  # if set, T sampled U(lo,hi) per sample
    cantera_phi_range: Optional[Tuple[float, float]] = None  # equivalence ratio range; None = fixed composition
    cantera_composition: str = 'CH4:1.0, O2:2.0, N2:7.52'
    cantera_use_mole_fractions: bool = False  # convert concentrations → mole fractions before storing
    default_tail: Optional[float] = None  # if set, overrides CLI --tail for this model

def _glycolysis_oracle22_config() -> ModelConfig:
    """
    Fixed-oracle glycolysis config.  The 33 parameters are set to a single
    ground-truth vector (lo == hi), so every trajectory shares the same kinetics
    and only the bolus pattern differs.

    Control channels (7): Glc (0), I_HK (16), I_PFK (17), I_GAPDH (18),
                          I_PGK (19), I_ENO (20), I_PK (21)
    Observed (4):         Glc, Pyr, ATP, NADH → indices [0, 10, 12, 15]

    Suggested CLI flags:
      --obs-indices 0,10,12,15  --control-indices 0,16,17,18,19,20,21
      --t-span 20  --n-steps 400
    """
    true_params = [
        # k_hk, k_pgi_f, k_pgi_r, k_pfk, k_ald_f, k_ald_r
        0.5,  2.0,  1.0,  0.5,  0.5,  0.1,
        # k_tpi_f, k_tpi_r, k_gapdh, k_pgk, k_pgm_f, k_pgm_r
        5.0,  2.0,  1.0,  2.0,  2.0,  1.0,
        # k_eno, k_pk, k_ldh, k_pyr_sink, k_atpase, k_nadh_ox
        1.0,  2.0,  0.5,  0.1,  0.2,  0.3,
        # k_leak_hex, k_leak_tri, k_leak_lower
        0.02, 0.02, 0.01,
        # K_I_HK, K_I_PFK, K_I_GAPDH, K_I_PGK, K_I_ENO, K_I_PK
        1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
        # k_I_HK_decay, k_I_PFK_decay, k_I_GAPDH_decay, k_I_PGK_decay, k_I_ENO_decay, k_I_PK_decay
        0.2,  0.2,  0.2,  0.2,  0.2,  0.2,
    ]
    param_ranges = [(v, v) for v in true_params]

    x0_default = [
        5.0,              # Glc
        0.1, 0.05, 0.02,  # G6P, F6P, FBP
        0.01, 0.01,       # GAP, DHAP
        0.0, 0.0, 0.0, 0.0,  # BPG13, PG3, PG2, PEP
        0.0, 0.0,         # Pyr, Lac
        2.0, 0.5,         # ATP, ADP
        1.0, 0.1,         # NAD, NADH
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # inhibitor pools start at zero
    ]

    bolus_ranges = {
        "Glc":    (0.5, 3.0),
        "I_HK":   (0.1, 2.0),
        "I_PFK":  (0.1, 2.0),
        "I_GAPDH": (0.1, 2.0),
        "I_PGK":  (0.1, 2.0),
        "I_ENO":  (0.1, 2.0),
        "I_PK":   (0.1, 2.0),
    }

    return ModelConfig(
        param_ranges=param_ranges,
        x0_default=x0_default,
        bolus_ranges=bolus_ranges,
        bolus_default=(0.1, 2.0),
        bolus_count_range=(3, 15),
        simulator="ivp",
    )


def _fixed_params_from_bounds(theta_lo_vec: list[float], theta_hi_vec: list[float]) -> list[float]:
    if len(theta_lo_vec) != len(theta_hi_vec):
        raise ValueError("theta_lo_vec and theta_hi_vec must have same length")
    return [0.5 * (float(lo) + float(hi)) for lo, hi in zip(theta_lo_vec, theta_hi_vec)]


def _glycolysis_reduced12_config() -> ModelConfig:
    scaffold = _get_glycolysis_reduced12()
    true_params = _fixed_params_from_bounds(scaffold.theta_lo_vec, scaffold.theta_hi_vec)
    param_ranges = [(v, v) for v in true_params]
    # States: Glc, HexP, TriP, BPG13, PG3, PEP, Pyr, Lac, ATP, ADP, NAD, NADH
    x0_default = [
        5.0,   # Glc
        0.1,   # HexP
        0.05,  # TriP
        0.0,   # BPG13
        0.0,   # PG3
        0.0,   # PEP
        0.0,   # Pyr
        0.0,   # Lac
        2.0,   # ATP
        0.5,   # ADP
        1.0,   # NAD
        0.1,   # NADH
    ]
    bolus_ranges = {
        "Glc": (0.5, 3.0),
        "ATP": (0.1, 2.0),
        "ADP": (0.1, 2.0),
        "NAD": (0.1, 2.0),
        "NADH": (0.1, 2.0),
    }
    return ModelConfig(
        param_ranges=param_ranges,
        x0_default=x0_default,
        bolus_ranges=bolus_ranges,
        bolus_default=(0.1, 2.0),
        bolus_count_range=(3, 15),
        simulator="ivp",
    )


def _glycolysis_reduced8_config() -> ModelConfig:
    scaffold = _get_glycolysis_reduced8()
    true_params = _fixed_params_from_bounds(scaffold.theta_lo_vec, scaffold.theta_hi_vec)
    param_ranges = [(v, v) for v in true_params]
    # States: Glc, SugarP, PEP, Pyr, Lac, ATP, NAD, NADH
    x0_default = [5.0, 0.1, 0.0, 0.0, 0.0, 2.0, 1.0, 0.1]
    bolus_ranges = {
        "Glc": (0.5, 3.0),
        "ATP": (0.1, 2.0),
        "NAD": (0.1, 2.0),
        "NADH": (0.1, 2.0),
    }
    return ModelConfig(
        param_ranges=param_ranges,
        x0_default=x0_default,
        bolus_ranges=bolus_ranges,
        bolus_default=(0.1, 2.0),
        bolus_count_range=(3, 15),
        simulator="ivp",
    )


def _glycolysis_reduced4_config() -> ModelConfig:
    scaffold = _get_glycolysis_reduced4()
    true_params = _fixed_params_from_bounds(scaffold.theta_lo_vec, scaffold.theta_hi_vec)
    param_ranges = [(v, v) for v in true_params]
    # States: Glc, Pyr, ATP, NADH
    x0_default = [5.0, 0.0, 2.0, 0.1]
    bolus_ranges = {
        "Glc": (0.5, 3.0),
        "ATP": (0.1, 2.0),
        "NADH": (0.1, 2.0),
    }
    return ModelConfig(
        param_ranges=param_ranges,
        x0_default=x0_default,
        bolus_ranges=bolus_ranges,
        bolus_default=(0.1, 2.0),
        bolus_count_range=(3, 15),
        simulator="ivp",
    )


def _aramco_30_config() -> ModelConfig:
    """
    AramcoMech 3.0 configuration using Cantera to fetch real kinetics.

    Hardness levers:
      cantera_T_range=(950, 1050): ignition delay varies ~0.5-5s across samples,
        forcing the model to learn rate-constant temperature dependence rather than
        memorizing a single ignition time.
      cantera_phi_range=(0.7, 1.3): lean/stoichiometric/rich combustion gives
        qualitatively different product distributions (CO2 vs CO, H2O vs H2).

    param_ranges are extracted at T=1000K but x0 is recomputed per-sample using
    the sampled T and phi — the rate constants don't change (fixed ground truth),
    only the initial concentrations and thus the trajectory shape.
    """
    gas = ct.Solution('datasets/methane/aramco_30.yaml')
    T_INIT = 1000.0
    COMPOSITION = 'CH4:1.0, O2:2.0, N2:7.52'
    gas.TPX = T_INIT, ct.one_atm, COMPOSITION

    real_k_params = gas.forward_rate_constants.tolist()
    param_ranges = [(v, v) for v in real_k_params]
    x0_default = gas.concentrations.tolist()  # fallback if no T/phi range used

    # Bolus amounts in kmol/m³ — ~5-20% of initial species concentration
    # These are applied before mole-fraction conversion so units are consistent.
    bolus_ranges = {
        "CH4": (0.0001, 0.0005),
        "O2":  (0.0002, 0.0010),
        "N2":  (0.0005, 0.0020),
        "H2O": (0.0001, 0.0005),
    }

    return ModelConfig(
        param_ranges=param_ranges,
        x0_default=x0_default,
        bolus_ranges=bolus_ranges,
        bolus_default=(0.0001, 0.0005),
        bolus_count_range=(2, 8),
        simulator="cantera",
        cantera_T=T_INIT,
        cantera_T_range=(950, 1050),
        cantera_phi_range=(0.7, 1.3),
        cantera_composition=COMPOSITION,
        cantera_use_mole_fractions=True,  # store X ∈ [0,1] instead of kmol/m³
        default_tail=3.5,
    )

def _mof_synthesis_config() -> ModelConfig:
    default_params = [5.0, 1.0, 3.0, 2.0, 0.5, 0.1, 10.0, 1.0, 1.0, 3.0,
                      0.5, 4.0, 1.0, 1.5, 1.0, 1.0]
    param_ranges = [(v, v) for v in default_params]
    x0_default = [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    bolus_ranges = {"Base": (0.1, 2.0), "Mod":  (0.1, 1.0)}
    return ModelConfig(
        param_ranges=param_ranges, x0_default=x0_default, bolus_ranges=bolus_ranges,
        bolus_default=(0.1, 1.0), bolus_count_range=(2, 8), simulator="ivp"
    )

def _single_enzyme_config() -> ModelConfig:
    default_params = [10.0, 2.0, 2.0, 2.0, 5.0, 5.0]
    param_ranges = [(v, v) for v in default_params]
    x0_default = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    bolus_ranges = {"A": (0.5, 5.0), "B": (0.5, 5.0)}
    return ModelConfig(
        param_ranges=param_ranges, x0_default=x0_default, bolus_ranges=bolus_ranges,
        bolus_default=(0.5, 5.0), bolus_count_range=(2, 8), simulator="ivp"
    )

def _gri30_full_config() -> ModelConfig:
    gas = ct.Solution('gri30.yaml')
    gas.TPX = 1500.0, ct.one_atm, 'CH4:1.0, O2:2.0, N2:7.52'
    real_k_params = gas.forward_rate_constants.tolist()
    param_ranges = [(v, v) for v in real_k_params]
    x0_default = [0.0] * 53
    x0_default[13] = 1.0   
    x0_default[3]  = 2.0   
    x0_default[47] = 7.52  
    bolus_ranges = {"CH4": (0.05, 0.5), "O2":  (0.1,  1.0), "N2":  (0.5,  3.0), "H2O": (0.1,  1.0)}
    return ModelConfig(
        param_ranges=param_ranges, x0_default=x0_default, bolus_ranges=bolus_ranges,
        bolus_default=(0.1, 1.0), bolus_count_range=(2, 8), simulator="ivp"
    )

def _kazakov_middle_config() -> ModelConfig:
    default_params = [1.0] * 116
    param_ranges = [(v, v) for v in default_params]
    x0_default = [0.0] * 28
    x0_default[11] = 1.0   
    x0_default[3]  = 2.0   
    x0_default[22] = 7.52  
    bolus_ranges = {"CH4": (0.05, 0.5), "O2":  (0.1,  1.0), "N2":  (0.5,  3.0), "H2O": (0.1,  1.0)}
    return ModelConfig(
        param_ranges=param_ranges, x0_default=x0_default, bolus_ranges=bolus_ranges,
        bolus_default=(0.1, 1.0), bolus_count_range=(2, 8), simulator="ivp"
    )

def _smooke_reduced_config() -> ModelConfig:
    default_params = [1.0] * 35
    param_ranges = [(v, v) for v in default_params]
    x0_default = [0.0] * 16
    x0_default[0]  = 1.0   
    x0_default[2]  = 2.0   
    x0_default[15] = 7.52  
    bolus_ranges = {"CH4": (0.05, 0.5), "O2":  (0.1,  1.0), "N2":  (0.5,  3.0), "H2O": (0.1,  1.0)}
    return ModelConfig(
        param_ranges=param_ranges, x0_default=x0_default, bolus_ranges=bolus_ranges,
        bolus_default=(0.1, 1.0), bolus_count_range=(2, 8), simulator="ivp"
    )

MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "full13":               ModelConfig(),
    "mof_synthesis":        _mof_synthesis_config(),
    "single_enzyme":        _single_enzyme_config(),
    "gri30_full":           _gri30_full_config(),
    "kazakov_middle":       _kazakov_middle_config(),
    "smooke_reduced_model": _smooke_reduced_config(),
    "aramco_30":            _aramco_30_config(),
    "glycolysis_oracle22":  _glycolysis_oracle22_config(),
    "glycolysis_reduced12": _glycolysis_reduced12_config(),
    "glycolysis_reduced8":  _glycolysis_reduced8_config(),
    "glycolysis_reduced4":  _glycolysis_reduced4_config(),
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
    *, rng: np.random.Generator, n_bolus: int, t_start: float, t_end: float,
    amount_range: Tuple[float, float], n_channels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_event = rng.uniform(t_start, t_end, size=n_bolus).astype(np.float32)
    t_event.sort()
    ch_event = rng.integers(0, n_channels, size=n_bolus, dtype=np.int64)
    amt_event = rng.uniform(amount_range[0], amount_range[1], size=n_bolus).astype(np.float32)
    return t_event, ch_event, amt_event

def _generate_random_events_per_channel(
    *, rng: np.random.Generator, n_bolus: int, t_start: float, t_end: float,
    n_channels: int, control_names: np.ndarray,
    bolus_ranges: Optional[Dict[str, Tuple[float, float]]], bolus_default: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    *, t_obs: np.ndarray, t_event: np.ndarray, ch_event: np.ndarray,
    amt_event: np.ndarray, d_in: int,
) -> np.ndarray:
    K = int(t_obs.shape[0] - 1)
    u_bins = np.zeros((K, d_in), dtype=np.float32)
    for t_bolus, ch, amt in zip(t_event, ch_event, amt_event):
        k = int(np.searchsorted(t_obs, np.float32(t_bolus), side="right")) - 1
        k = max(0, min(k, K - 1))
        u_bins[k, int(ch)] += np.float32(amt)
    return u_bins

def _sample_params(
    rng: np.random.Generator, n_params: int, param_ranges: Optional[List[Tuple[float, float]]],
) -> np.ndarray:
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

def _build_x0(n_states_full: int, zero_init: bool, x0_default: Optional[List[float]]) -> np.ndarray:
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
    if output_file is None:
        raise ValueError("output_file must be provided")
    if model_fn is None:
        model_fn = FullModel

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _dim = model_fn(None, None, None, dim=True)
    n_states_full, n_params_full, names_full = _dim[0], _dim[1], _dim[2]
    names_full = list(names_full)

    mcfg = MODEL_CONFIGS.get(model_name, ModelConfig())

    if obs_indices is None:
        obs_indices = [0, 3, 6, 9, 12]  
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

    theta_true = _sample_params(rng, n_params_full, mcfg.param_ranges)

    theta_full = None
    if k_noise > 0.0:
        print(f"  adding kinetics noise with std={k_noise}")
        theta_full = np.repeat(theta_true[None, :], n_samples, axis=0).astype(np.float32)

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

        effective_tail = mcfg.default_tail if mcfg.default_tail is not None else tail
        if mcfg.bolus_ranges is not None:
            t_event, ch_event, amt_event = _generate_random_events_per_channel(
                rng=rng, n_bolus=n_bolus, t_start=0.0, t_end=max(0.0, t_span - effective_tail),
                n_channels=d_in, control_names=control_names,
                bolus_ranges=mcfg.bolus_ranges, bolus_default=mcfg.bolus_default,
            )
        else:
            t_event, ch_event, amt_event = _generate_random_events_idx(
                rng=rng, n_bolus=n_bolus, t_start=0.0, t_end=max(0.0, t_span - effective_tail),
                amount_range=mcfg.bolus_default, n_channels=d_in,
            )

        events = [(float(t), str(control_names[int(ch)]), float(a)) for t, ch, a in zip(t_event, ch_event, amt_event)]
        x0_full = x0_template.copy()

        try:
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
            elif mcfg.simulator == "cantera":
                gas_obj = ct.Solution('datasets/methane/aramco_30.yaml')

                # Sample temperature and equivalence ratio per trajectory
                T_sample = float(rng.uniform(*mcfg.cantera_T_range)) if mcfg.cantera_T_range else mcfg.cantera_T
                if mcfg.cantera_phi_range is not None:
                    phi = float(rng.uniform(*mcfg.cantera_phi_range))
                    # phi=1 is stoichiometric CH4+2O2; scale O2 and N2 by 1/phi
                    o2 = 2.0 / phi
                    composition = f'CH4:1.0, O2:{o2:.4f}, N2:{3.76*o2:.4f}'
                else:
                    composition = mcfg.cantera_composition
                gas_obj.TPX = T_sample, ct.one_atm, composition
                # x0 must match the sampled state, not the config default
                x0_full = gas_obj.concentrations.copy()

                t_solver, x_solver = simulate_cantera_with_bolus(
                    gas_object=gas_obj,
                    y0=x0_full,
                    t_start=0.0,
                    t_end=t_span,
                    bolus_gen=single_event_generator(events),
                    species_names=names_full,
                    t_record=t_obs,
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
        except Exception:
            n_failed += 1
            y0[i] = 0.0
            y_seq[i] = 0.0
            u_seq[i] = 0.0
            continue

        x_solver = np.asarray(x_solver, dtype=np.float32)
        t_solver = np.asarray(t_solver, dtype=np.float32)

        # Convert kmol/m³ → mole fractions for cantera outputs so values are in [0,1]
        # This makes log1p normalization behave correctly during training.
        if mcfg.cantera_use_mole_fractions:
            c_total = x_solver.sum(axis=1, keepdims=True).clip(min=1e-30)
            x_solver = x_solver / c_total

        if np.any(np.isnan(x_solver)) or np.any(x_solver < -100):
            n_failed += 1
            y0[i] = 0.0
            y_seq[i] = 0.0
            u_seq[i] = 0.0
            continue

        x_grid = interp1d(t_solver, x_solver, axis=0, kind="linear", fill_value="extrapolate")(t_obs).astype(np.float32)

        y_grid = x_grid[:, obs_indices]
        y0[i] = y_grid[0]
        y_seq[i] = y_grid[1:]

        u_seq[i] = _bin_events_to_u_seq(
            t_obs=t_obs, t_event=t_event, ch_event=ch_event, amt_event=amt_event, d_in=d_in,
        )

    if n_failed > 0:
        print(f"WARNING: {n_failed}/{n_samples} samples had NaN or large negatives (zeroed out)")

    save_kwargs = dict(
        y0=y0, u_seq=u_seq, y_seq=y_seq, t_obs=t_obs,
        control_indices=control_indices, obs_indices=obs_indices,
        names_full=np.asarray(names_full, dtype="<U16"),
        control_names=control_names, obs_names=obs_names,
        n_states_full=np.int64(n_states_full), n_params_full=np.int64(n_params_full),
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
                        choices=sorted(SIM_MODELS.keys()),
                        help="Simulation model to use for data generation.")
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
    suffix = f"N{args.n_samples}_T{int(args.t_span)}_steps{args.n_steps}_{init_tag}_knoise{args.k_noise}"

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