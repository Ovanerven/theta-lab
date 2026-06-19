import torch
import torch.nn as nn
import math
from typing import Dict, List

class MechanisticScaffold(nn.Module):
    """Base class for mechanistic scaffolds.

    Carries optional hooks (precompute_batch / initial_state / analytic_step /
    emit_theta) used by models that support analytic-step integration. The base
    implementations are scriptable no-ops so non-analytic scaffolds compile fine
    even when the model code references the hooks behind a constant gate.
    """

    # `__constants__` lets TorchScript treat these as compile-time so models
    # that gate `if scaffold.has_analytic_step:` can DCE the dead branch.
    __constants__ = ["has_analytic_step", "tf_at_k_zero", "theta_dim_emit"]

    def __init__(self, P: int, theta_dim: int):
        super().__init__()
        self.P = int(P)
        self.theta_dim = int(theta_dim)
        # Encoder emits theta_dim params; some scaffolds repack to a different
        # width via emit_theta() before the loss sees it (e.g. an IVTT loss).
        self.theta_dim_emit = int(theta_dim)
        # Set as instance attrs (so TorchScript tracks them); subclasses overwrite.
        self.has_analytic_step: bool = False
        self.tf_at_k_zero: bool = False
        self.state_names: List[str] = []
        # Per-parameter bounds — set by subclasses. None means use scalar fallback.
        self.theta_lo_vec: "list[float] | None" = None
        self.theta_hi_vec: "list[float] | None" = None

        # Partial-observability hooks — set by subclasses when scaffold.P differs
        # from dataset.P_obs (e.g. Model 3 has P=3 but the dataset carries P=7
        # columns; mm/pm sit at scaffold state indices 0/1, not 3/5).
        # None = identity mapping (assume every state is observed in scaffold order).
        #
        # `obs_state_idx[j]` = scaffold state index that maps to the j-th
        # observed column in the dataset's y_seq (in dataset order).
        # `control_state_map` = {control_name: scaffold_state_index} for bolus
        # additions; the trainer rebuilds u_to_y_jump from this.
        self.obs_state_idx: "list[int] | None" = None
        self.control_state_map: "dict[str, int] | None" = None

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ---------- analytic-scaffold hooks (used iff has_analytic_step) ----------
    # All four are scriptable no-ops on the base class so TorchScript can compile
    # any model that calls `self.rhs.<hook>(...)` regardless of which concrete
    # scaffold is used. Subclasses override with real bodies.

    def precompute_batch(
        self, y0: torch.Tensor, u_seq: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        return out

    def initial_state(self, y0: torch.Tensor) -> torch.Tensor:
        return y0

    def analytic_step(
        self,
        y_prev: torch.Tensor,
        dt_k: torch.Tensor,
        theta_k: torch.Tensor,
        ctx: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # No-op default — subclasses with has_analytic_step=True must override.
        return y_prev

    def emit_theta(
        self, theta_enc: torch.Tensor, y_state: torch.Tensor,
    ) -> torch.Tensor:
        return theta_enc
# ── Synthetic benchmark scaffolds (MOF synthesis, single-enzyme kinetics) ──
# Ground-truth data for these is generated via create_dataset.py + sim/.
# Glycolysis scaffolds live in sim/glycolysis.py (imported above).

class MOFSynthesis12Scaffold(MechanisticScaffold):
    """
    Full 12-state MOF synthesis scaffold. Preserves all mechanistic structure
    from MOF_model.py; all 16 kinetic constants are learned as θ(t).

    States (12): Met, LigH, Lig_minus, H_plus, Base, Mod,
                 SBU, SBU_capped, Nuc_A, Am, Nuc_C, MOF_C
    Control inputs (bolused): Base (idx 4), Mod (idx 5)

    Parameters θ (16):
      0  k_deprot  : LigH + Base -> Lig_minus deprotonation rate
      1  k_prot    : Lig_minus + H+ -> LigH reprotonation rate
      2  k_oli     : Met^a * Lig_minus^b -> SBU oligomerization rate
      3  k_cap     : SBU + Mod -> SBU_capped capping rate
      4  k_uncap   : SBU_capped -> SBU + Mod uncapping rate
      5  K_I       : modulator inhibition constant for crystalline growth
      6  knuc_A    : amorphous nucleation prefactor
      7  kgro_A    : amorphous growth rate
      8  kagg_A    : amorphous aggregation rate
      9  n_A       : SBU exponent for amorphous nucleation
      10 knuc_C    : crystalline nucleation prefactor
      11 kgro_C    : crystalline growth rate
      12 kagg_C    : crystalline aggregation rate
      13 n_C       : SBU exponent for crystalline nucleation
      14 a         : Met exponent in oligomerization
      15 b         : Lig_minus exponent in oligomerization
    """
    def __init__(self):
        super().__init__(P=12, theta_dim=16)
        self.state_names = [
            "Met", "LigH", "Lig_minus", "H_plus",
            "Base", "Mod", "SBU", "SBU_capped",
            "Nuc_A", "Am", "Nuc_C", "MOF_C",
        ]
        # Per-parameter bounds (true values: k_deprot=5, k_prot=1, k_oli=3, k_cap=2,
        # k_uncap=0.5, K_I=0.1, knuc_A=10, kgro_A=1, kagg_A=1, n_A=3,
        # knuc_C=0.5, kgro_C=4, kagg_C=1, n_C=1.5, a=1, b=1)
        self.theta_lo_vec = [0.1,  0.01, 0.01, 0.01, 0.001, 0.001,
                             0.1,  0.01, 0.01, 0.5,
                             0.001, 0.01, 0.01, 0.5,
                             0.1, 0.1]
        self.theta_hi_vec = [50.0, 20.0, 30.0, 20.0, 10.0, 2.0,
                             100.0, 20.0, 20.0, 10.0,
                             20.0, 50.0, 20.0, 8.0,
                             5.0, 5.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        (
            Met, LigH, Lig_minus, H_plus,
            Base, Mod, SBU, SBU_capped,
            Nuc_A, Am, Nuc_C, MOF_C,
        ) = y.unbind(dim=-1)
        (
            k_deprot, k_prot, k_oli, k_cap, k_uncap, K_I,
            knuc_A, kgro_A, kagg_A, n_A,
            knuc_C, kgro_C, kagg_C, n_C,
            a, b,
        ) = theta.unbind(dim=-1)

        Met_p         = torch.clamp_min(Met, 0.0)
        LigH_p        = torch.clamp_min(LigH, 0.0)
        Lig_minus_p   = torch.clamp_min(Lig_minus, 0.0)
        H_plus_p      = torch.clamp_min(H_plus, 0.0)
        Base_p        = torch.clamp_min(Base, 0.0)
        Mod_p         = torch.clamp_min(Mod, 0.0)
        SBU_p         = torch.clamp_min(SBU, 0.0)
        SBU_capped_p  = torch.clamp_min(SBU_capped, 0.0)
        Nuc_A_p       = torch.clamp_min(Nuc_A, 0.0)
        Am_p          = torch.clamp_min(Am, 0.0)
        Nuc_C_p       = torch.clamp_min(Nuc_C, 0.0)
        MOF_C_p       = torch.clamp_min(MOF_C, 0.0)

        k_deprot = torch.clamp_min(k_deprot, 0.0)
        k_prot   = torch.clamp_min(k_prot,   0.0)
        k_oli    = torch.clamp_min(k_oli,    0.0)
        k_cap    = torch.clamp_min(k_cap,    0.0)
        k_uncap  = torch.clamp_min(k_uncap,  0.0)
        K_I      = torch.clamp_min(K_I,      1e-6)
        knuc_A   = torch.clamp_min(knuc_A,   0.0)
        kgro_A   = torch.clamp_min(kgro_A,   0.0)
        kagg_A   = torch.clamp_min(kagg_A,   0.0)
        n_A      = torch.clamp_min(n_A,      1e-6)
        knuc_C   = torch.clamp_min(knuc_C,   0.0)
        kgro_C   = torch.clamp_min(kgro_C,   0.0)
        kagg_C   = torch.clamp_min(kagg_C,   0.0)
        n_C      = torch.clamp_min(n_C,      1e-6)
        a        = torch.clamp_min(a,        1e-6)
        b        = torch.clamp_min(b,        1e-6)

        r_deprot = k_deprot * LigH_p * Base_p
        r_prot   = k_prot * Lig_minus_p * H_plus_p
        r_oli    = k_oli * (Met_p + 1e-8).pow(a) * (Lig_minus_p + 1e-8).pow(b)
        r_cap    = k_cap * SBU_p * Mod_p
        r_uncap  = k_uncap * SBU_capped_p
        r_nuc_A  = knuc_A * (SBU_p + 1e-8).pow(n_A)
        r_nuc_C  = knuc_C * (SBU_p + 1e-8).pow(n_C)
        r_gro_A  = kgro_A * SBU_p * Am_p
        r_agg_A  = kagg_A * Nuc_A_p.pow(2.0)
        inhib    = K_I / (K_I + Mod_p + 1e-6)
        r_gro_C  = kgro_C * SBU_p * MOF_C_p * inhib
        r_agg_C  = kagg_C * Nuc_C_p.pow(2.0)

        dMet        = -r_oli
        dLigH       = -r_deprot + r_prot
        dLig_minus  =  r_deprot - r_prot - r_oli
        dH_plus     =  r_deprot - r_prot + r_oli
        dBase       = -r_deprot
        dMod        = -r_cap + r_uncap
        dSBU        =  r_oli - r_cap + r_uncap - r_nuc_A - r_gro_A - r_nuc_C - r_gro_C
        dSBU_capped =  r_cap - r_uncap
        dNuc_A      =  r_nuc_A - r_agg_A
        dAm         =  r_agg_A + r_gro_A
        dNuc_C      =  r_nuc_C - r_agg_C
        dMOF_C      =  r_agg_C + r_gro_C

        return torch.stack((
            dMet, dLigH, dLig_minus, dH_plus,
            dBase, dMod, dSBU, dSBU_capped,
            dNuc_A, dAm, dNuc_C, dMOF_C,
        ), dim=-1)


class MOFSynthesis8Scaffold(MechanisticScaffold):
    """
    8-state MOF synthesis scaffold. Collapses the four deprotonation species
    (Met, LigH, Lig_minus, H_plus) into an effective SBU production term driven
    by Base. Retains SBU_capped explicitly so Mod dynamics are exact. Includes
    cooperative nucleation exponents n_A, n_C as learned θ parameters.

    States (8): Base, Mod, SBU, SBU_capped, Nuc_A, Am, Nuc_C, MOF_C
    Control inputs (bolused): Base (idx 0), Mod (idx 1)

    Parameters θ (13):
      0  k_base_decay : effective Base consumption rate
      1  k_oli_eff    : effective SBU production rate from Base
      2  k_cap        : SBU + Mod -> SBU_capped capping rate
      3  k_uncap      : SBU_capped -> SBU + Mod uncapping rate
      4  K_I          : modulator inhibition constant
      5  knuc_A       : amorphous nucleation prefactor
      6  kgro_A       : amorphous growth rate
      7  kagg_A       : amorphous aggregation rate
      8  n_A          : SBU exponent for amorphous nucleation
      9  knuc_C       : crystalline nucleation prefactor
      10 kgro_C       : crystalline growth rate
      11 kagg_C       : crystalline aggregation rate
      12 n_C          : SBU exponent for crystalline nucleation
    """
    def __init__(self):
        super().__init__(P=8, theta_dim=13)
        self.state_names = [
            "Base", "Mod", "SBU", "SBU_capped",
            "Nuc_A", "Am", "Nuc_C", "MOF_C",
        ]
        # Per-parameter bounds (k_base_decay, k_oli_eff, k_cap, k_uncap, K_I,
        # knuc_A, kgro_A, kagg_A, n_A, knuc_C, kgro_C, kagg_C, n_C)
        self.theta_lo_vec = [0.1,  0.01, 0.01, 0.001, 0.001,
                             0.1,  0.01, 0.01, 0.5,
                             0.001, 0.01, 0.01, 0.5]
        self.theta_hi_vec = [50.0, 30.0, 20.0, 10.0,  2.0,
                             100.0, 20.0, 20.0, 10.0,
                             20.0,  50.0, 20.0, 8.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        Base, Mod, SBU, SBU_capped, Nuc_A, Am, Nuc_C, MOF_C = y.unbind(dim=-1)
        (
            k_base_decay, k_oli_eff, k_cap, k_uncap, K_I,
            knuc_A, kgro_A, kagg_A, n_A,
            knuc_C, kgro_C, kagg_C, n_C,
        ) = theta.unbind(dim=-1)

        Base_p        = torch.clamp_min(Base, 0.0)
        Mod_p         = torch.clamp_min(Mod, 0.0)
        SBU_p         = torch.clamp_min(SBU, 0.0)
        SBU_capped_p  = torch.clamp_min(SBU_capped, 0.0)
        Nuc_A_p       = torch.clamp_min(Nuc_A, 0.0)
        Am_p          = torch.clamp_min(Am, 0.0)
        Nuc_C_p       = torch.clamp_min(Nuc_C, 0.0)
        MOF_C_p       = torch.clamp_min(MOF_C, 0.0)

        k_base_decay = torch.clamp_min(k_base_decay, 0.0)
        k_oli_eff    = torch.clamp_min(k_oli_eff,    0.0)
        k_cap        = torch.clamp_min(k_cap,        0.0)
        k_uncap      = torch.clamp_min(k_uncap,      0.0)
        K_I          = torch.clamp_min(K_I,          1e-6)
        knuc_A       = torch.clamp_min(knuc_A,       0.0)
        kgro_A       = torch.clamp_min(kgro_A,       0.0)
        kagg_A       = torch.clamp_min(kagg_A,       0.0)
        n_A          = torch.clamp_min(n_A,          1e-6)
        knuc_C       = torch.clamp_min(knuc_C,       0.0)
        kgro_C       = torch.clamp_min(kgro_C,       0.0)
        kagg_C       = torch.clamp_min(kagg_C,       0.0)
        n_C          = torch.clamp_min(n_C,          1e-6)

        r_cap    = k_cap * SBU_p * Mod_p
        r_uncap  = k_uncap * SBU_capped_p
        r_nuc_A  = knuc_A * (SBU_p + 1e-8).pow(n_A)
        r_nuc_C  = knuc_C * (SBU_p + 1e-8).pow(n_C)
        r_gro_A  = kgro_A * SBU_p * Am_p
        r_agg_A  = kagg_A * Nuc_A_p.pow(2.0)
        inhib    = K_I / (K_I + Mod_p + 1e-6)
        r_gro_C  = kgro_C * SBU_p * MOF_C_p * inhib
        r_agg_C  = kagg_C * Nuc_C_p.pow(2.0)

        dBase       = -k_base_decay * Base_p
        dMod        = -r_cap + r_uncap
        dSBU        =  k_oli_eff * Base_p - r_cap + r_uncap - r_nuc_A - r_gro_A - r_nuc_C - r_gro_C
        dSBU_capped =  r_cap - r_uncap
        dNuc_A      =  r_nuc_A - r_agg_A
        dAm         =  r_agg_A + r_gro_A
        dNuc_C      =  r_nuc_C - r_agg_C
        dMOF_C      =  r_agg_C + r_gro_C

        return torch.stack((
            dBase, dMod, dSBU, dSBU_capped,
            dNuc_A, dAm, dNuc_C, dMOF_C,
        ), dim=-1)


class MOFSynthesis6Scaffold(MechanisticScaffold):
    """
    6-state MOF synthesis scaffold. Applies two further reductions on top of
    MOFSynthesis8Scaffold: (1) quasi-steady-state on SBU_capped so dMod = 0
    between boluses (net capping flux is zero); (2) fast-nucleation collapse of
    Nuc_A directly into Am. Retains cooperative nucleation exponents n_A, n_C
    as learned θ parameters (advisor recommendation: option b).

    States (6): Base, Mod, SBU, Am, Nuc_C, MOF_C
    Control inputs (bolused): Base (idx 0), Mod (idx 1)

    Parameters θ (10):
      0  k_base_decay : effective Base consumption rate
      1  k_oli_eff    : effective SBU production rate from Base
      2  knuc_A       : amorphous nucleation prefactor (feeds Am directly)
      3  kgro_A       : amorphous growth rate
      4  n_A          : SBU exponent for amorphous nucleation
      5  knuc_C       : crystalline nucleation prefactor
      6  kgro_C       : crystalline growth rate
      7  kagg_C       : crystalline aggregation rate
      8  n_C          : SBU exponent for crystalline nucleation
      9  K_I          : modulator inhibition constant
    """
    def __init__(self):
        super().__init__(P=6, theta_dim=10)
        self.state_names = ["Base", "Mod", "SBU", "Am", "Nuc_C", "MOF_C"]
        # Per-parameter bounds (k_base_decay, k_oli_eff, knuc_A, kgro_A, n_A,
        # knuc_C, kgro_C, kagg_C, n_C, K_I)
        self.theta_lo_vec = [0.1,  0.01, 0.1,  0.01, 0.5,  0.001, 0.01, 0.01, 0.5,  0.001]
        self.theta_hi_vec = [50.0, 30.0, 100.0, 20.0, 10.0, 20.0, 50.0, 20.0, 8.0,  2.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        Base, Mod, SBU, Am, Nuc_C, MOF_C = y.unbind(dim=-1)
        (
            k_base_decay, k_oli_eff,
            knuc_A, kgro_A, n_A,
            knuc_C, kgro_C, kagg_C, n_C,
            K_I,
        ) = theta.unbind(dim=-1)

        Base_p  = torch.clamp_min(Base,  0.0)
        Mod_p   = torch.clamp_min(Mod,   0.0)
        SBU_p   = torch.clamp_min(SBU,   0.0)
        Am_p    = torch.clamp_min(Am,    0.0)
        Nuc_C_p = torch.clamp_min(Nuc_C, 0.0)
        MOF_C_p = torch.clamp_min(MOF_C, 0.0)

        k_base_decay = torch.clamp_min(k_base_decay, 0.0)
        k_oli_eff    = torch.clamp_min(k_oli_eff,    0.0)
        knuc_A       = torch.clamp_min(knuc_A,       0.0)
        kgro_A       = torch.clamp_min(kgro_A,       0.0)
        n_A          = torch.clamp_min(n_A,          1e-6)
        knuc_C       = torch.clamp_min(knuc_C,       0.0)
        kgro_C       = torch.clamp_min(kgro_C,       0.0)
        kagg_C       = torch.clamp_min(kagg_C,       0.0)
        n_C          = torch.clamp_min(n_C,          1e-6)
        K_I          = torch.clamp_min(K_I,          1e-6)

        r_nuc_A  = knuc_A * (SBU_p + 1e-8).pow(n_A)
        r_nuc_C  = knuc_C * (SBU_p + 1e-8).pow(n_C)
        r_gro_A  = kgro_A * SBU_p * Am_p
        inhib    = K_I / (K_I + Mod_p + 1e-6)
        r_gro_C  = kgro_C * SBU_p * MOF_C_p * inhib
        r_agg_C  = kagg_C * Nuc_C_p.pow(2.0)

        dBase  = -k_base_decay * Base_p
        dMod   = torch.zeros_like(Base)   # QSS: r_cap == r_uncap between boluses
        dSBU   =  k_oli_eff * Base_p - r_nuc_A - r_gro_A - r_nuc_C - r_gro_C
        dAm    =  r_nuc_A + r_gro_A       # Nuc_A fast: collapses directly into Am
        dNuc_C =  r_nuc_C - r_agg_C
        dMOF_C =  r_agg_C + r_gro_C

        return torch.stack((dBase, dMod, dSBU, dAm, dNuc_C, dMOF_C), dim=-1)


class MOFSynthesis4Scaffold(MechanisticScaffold):
    """
    4-state MOF synthesis scaffold. Most aggressively reduced: no SBU tracked.
    Base acts as proxy for SBU availability; nucleation is linear (no cooperative
    exponent) since SBU is not an explicit state. Mod decays via a slow first-order
    approximation (GRU compensates for the full capping dynamics).

    States (4): Base, Mod, Am, MOF_C
    Control inputs (bolused): Base (idx 0), Mod (idx 1)

    Parameters θ (7):
      0  k_base   : effective Base decay rate
      1  k_mod    : effective Mod decay rate (first-order approximation)
      2  k_nuc_A  : amorphous nucleation rate (linear in Base)
      3  k_gro_A  : amorphous growth rate (Base * Am)
      4  k_nuc_C  : crystalline nucleation rate (linear in Base)
      5  k_gro_C  : crystalline growth rate (Base * MOF_C * inhibition)
      6  K_I      : modulator inhibition constant
    """
    def __init__(self):
        super().__init__(P=4, theta_dim=7)
        self.state_names = ["Base", "Mod", "Am", "MOF_C"]
        # Per-parameter bounds (k_base, k_mod, k_nuc_A, k_gro_A, k_nuc_C, k_gro_C, K_I)
        self.theta_lo_vec = [0.1,  0.001, 0.1,  0.01, 0.001, 0.01, 0.001]
        self.theta_hi_vec = [50.0, 10.0, 100.0, 20.0, 20.0,  50.0, 2.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        Base, Mod, Am, MOF_C = y.unbind(dim=-1)
        k_base, k_mod, k_nuc_A, k_gro_A, k_nuc_C, k_gro_C, K_I = theta.unbind(dim=-1)

        Base_p  = torch.clamp_min(Base,  0.0)
        Mod_p   = torch.clamp_min(Mod,   0.0)
        Am_p    = torch.clamp_min(Am,    0.0)
        MOF_C_p = torch.clamp_min(MOF_C, 0.0)

        k_base  = torch.clamp_min(k_base,  0.0)
        k_mod   = torch.clamp_min(k_mod,   0.0)
        k_nuc_A = torch.clamp_min(k_nuc_A, 0.0)
        k_gro_A = torch.clamp_min(k_gro_A, 0.0)
        k_nuc_C = torch.clamp_min(k_nuc_C, 0.0)
        k_gro_C = torch.clamp_min(k_gro_C, 0.0)
        K_I     = torch.clamp_min(K_I,     1e-6)

        inhib  = K_I / (K_I + Mod_p + 1e-6)

        dBase  = -k_base * Base_p
        dMod   = -k_mod * Mod_p
        dAm    =  k_nuc_A * Base_p + k_gro_A * Base_p * Am_p
        dMOF_C =  k_nuc_C * Base_p + k_gro_C * Base_p * MOF_C_p * inhib

        return torch.stack((dBase, dMod, dAm, dMOF_C), dim=-1)


class SingleEnzymeLumpedScaffold(MechanisticScaffold):
    """
    2-state reduced scaffold for the Single Enzyme scenario.

    The full 6-state system is simulated but only A (substrate, idx 0) and
    C (product, idx 2) are observed. The scaffold approximates the dynamics
    with a simple first-order reversible reaction:

        dA_approx = -kf * A + kr * C
        dC_approx =  kf * A - kr * C

    This is structurally wrong in two ways:
      1. The true reaction is bimolecular (rate ∝ A·B); B is hidden
      2. There is no saturation / denominator term

    The neural network must learn time-varying kf(t) and kr(t) to compensate
    for the missing B dependence and the wrong kinetics.

    States (2): S ↔ A (observed substrate), P ↔ C (observed product)
    Control: A-bolus maps to S; B-bolus is a hidden input (seen by the GRU
             via u_seq but not directly reflected in the observed state)
    Parameters θ (2): kf (effective forward rate), kr (effective reverse rate)

    Use with: datasets/single_enzyme_lumped.npz  (--obs-indices 0,2)
    """
    def __init__(self):
        super().__init__(P=2, theta_dim=2)
        self.state_names = ["S", "P"]
        # True rates are kcat_f·E ≈ 10 and kcat_r·E ≈ 2, but with the denominator
        # the effective observed rate is much lower; use wide bounds.
        self.theta_lo_vec = [0.001, 0.001]
        self.theta_hi_vec = [100.0,  50.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        S, P = y.unbind(dim=-1)
        kf, kr = theta.unbind(dim=-1)

        S_p = torch.clamp_min(S, 0.0)
        P_p = torch.clamp_min(P, 0.0)
        kf  = torch.clamp_min(kf, 0.0)
        kr  = torch.clamp_min(kr, 0.0)

        v = kf * S_p - kr * P_p

        dS = -v
        dP =  v

        return torch.stack((dS, dP), dim=-1)


class SingleEnzymeReduced4Scaffold(MechanisticScaffold):
    """
    Reduced 4-state mass-action scaffold for the Single Enzyme scenario.

    The true system uses Reversible Bi-Bi (Michaelis-Menten) kinetics with a
    nonlinear denominator. This scaffold intentionally simplifies to plain
    mass-action, dropping the inert states E and I (which are constant in the
    data: E=1, I=0) and removing the denominator entirely:

        v  = kf * A * B  −  kr * C * D

    The scaffold structure (A+B → C+D reversibly) is topologically correct,
    but the kinetics are wrong. The neural network must learn time-varying
    kf(t) and kr(t) to compensate for the missing saturation terms.

    States (4): A, B, C, D
    Control inputs (bolused): A (idx 0), B (idx 1)
    Parameters θ (2): kf (effective forward rate), kr (effective reverse rate)

    Use with: datasets/single_enzyme_4.npz  (--obs-indices 0,1,2,3)
    Ground-truth Bi-Bi values for reference: kcat_f·E=10, kcat_r·E=2
    """
    def __init__(self):
        super().__init__(P=4, theta_dim=2)
        self.state_names = ["A", "B", "C", "D"]
        # Bounds: true effective forward rate ≈ 10, reverse ≈ 2
        self.theta_lo_vec = [0.01, 0.001]
        self.theta_hi_vec = [200.0, 100.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, B, C, D = y.unbind(dim=-1)
        kf, kr = theta.unbind(dim=-1)

        A_p = torch.clamp_min(A, 0.0)
        B_p = torch.clamp_min(B, 0.0)
        C_p = torch.clamp_min(C, 0.0)
        D_p = torch.clamp_min(D, 0.0)

        kf = torch.clamp_min(kf, 0.0)
        kr = torch.clamp_min(kr, 0.0)

        v = kf * A_p * B_p - kr * C_p * D_p

        dA = -v
        dB = -v
        dC =  v
        dD =  v

        return torch.stack((dA, dB, dC, dD), dim=-1)


class SingleEnzymeScaffold(MechanisticScaffold):
    """
    6-state Reversible Bi-Bi enzyme kinetics scaffold.

    Reaction: A + B <-> C + D  (catalysed by enzyme E, inhibitor I inert)

    States (6): A, B, C, D, E, I
    Control inputs (bolused): A (idx 0), B (idx 1)

    Parameters θ (6):
      0  kcat_f : forward catalytic rate constant
      1  kcat_r : reverse catalytic rate constant
      2  Ka     : Michaelis constant for substrate A
      3  Kb     : Michaelis constant for substrate B
      4  Kc     : Michaelis constant for product C
      5  Kd     : Michaelis constant for product D

    Ground-truth values: kcat_f=10.0, kcat_r=2.0, Ka=2.0, Kb=2.0, Kc=5.0, Kd=5.0
    Dataset: datasets/single_enzyme_6.npz  (--t-span 10 --n-steps 200)
    """
    def __init__(self):
        super().__init__(P=6, theta_dim=6)
        self.state_names = ["A", "B", "C", "D", "E", "I"]
        # Per-parameter bounds: wide enough to contain the true values with room to search
        self.theta_lo_vec = [0.1,  0.01, 0.01, 0.01, 0.01, 0.01]
        self.theta_hi_vec = [100.0, 50.0, 50.0, 50.0, 50.0, 50.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, B, C, D, E, I = y.unbind(dim=-1)
        kcat_f, kcat_r, Ka, Kb, Kc, Kd = theta.unbind(dim=-1)

        eps: float = 1e-12

        A_p = torch.clamp_min(A, 0.0)
        B_p = torch.clamp_min(B, 0.0)
        C_p = torch.clamp_min(C, 0.0)
        D_p = torch.clamp_min(D, 0.0)
        E_p = torch.clamp_min(E, 0.0)

        Ka = torch.clamp_min(Ka, eps)
        Kb = torch.clamp_min(Kb, eps)
        Kc = torch.clamp_min(Kc, eps)
        Kd = torch.clamp_min(Kd, eps)

        Vf = kcat_f * E_p
        Vr = kcat_r * E_p

        D0 = Ka * Kb
        denom = (
            D0 * (1.0 + C_p / Kc + D_p / Kd + (C_p * D_p) / (Kc * Kd))
            + (Kb * A_p) * (1.0 + D_p / Kd)
            + (Ka * B_p) * (1.0 + C_p / Kc)
            + (A_p * B_p)
            + eps
        )

        v = (Vf * A_p * B_p - Vr * C_p * D_p) / denom

        dA = -v
        dB = -v
        dC =  v
        dD =  v
        dE = E * 0.0   # conserved: always zero
        dI = I * 0.0   # inert: always zero

        return torch.stack((dA, dB, dC, dD, dE, dI), dim=-1)

# -----------------------------------------------------------------------------
# 3) JIT‐scripted analytic ODE: This is the simplest model This has to be integrated into same format as the rest of the scaffolds here. 
# -----------------------------------------------------------------------------
# @torch.jit.script
# def _step_integration(
#     m0: torch.Tensor, p0: torch.Tensor,
#     dt: torch.Tensor,
#     VTX: torch.Tensor, KTX: torch.Tensor,
#     dna: torch.Tensor, kdm: torch.Tensor,
#     VTL: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     eps = 1e-8
#     A     = VTX * dna / (KTX + dna + eps)
#     m_inf = A / (kdm + eps)
#     expT  = torch.exp(-kdm * dt)
#     m1    = m_inf + (m0 - m_inf) * expT
#     int_m = m_inf * dt + (m0 - m_inf) * (1.0 - expT) / (kdm + eps)
#     p1    = p0 + VTL * int_m
#     return m1.clamp(min=0.0), p1.clamp(min=0.0), int_m


# class TXTL_mRNAMaturation(MechanisticScaffold):
#     # states:    [R, m, mm, p, pm]  (optionally +DNA as 6th state)
#     # theta:     [lam, VTXmax, kdm, VTLmax, kmt, kmatm]
#     def __init__(self):
#         super().__init__(P=5, theta_dim=6)
#         self.state_names = ["R", "m", "mm", "p", "pm"]
#         self.theta_lo_vec = [1e-6, 3e-5, 1e-5, 3e-5, 1e-5, 5e-5]
#         self.theta_hi_vec = [5e-4, 1.2e-1, 1e-2, 8e-2, 3.5e-4, 3.5e-3]

#     def forward(self, y, theta, dna):  # or embed DNA as y[:,5]
#         R, m, mm, p, pm = y.unbind(-1)
#         lam, VTXmax, kdm, VTLmax, kmt, kmatm = theta.unbind(-1)
#         dR  = -lam * R
#         dm  = R * VTXmax * dna - (kdm + kmatm) * m
#         dmm = kmatm * m - kdm * mm
#         dp  = R * VTLmax * (m + mm) - kmt * p
#         dpm = kmt * p
#         return torch.stack([dR, dm, dmm, dp, dpm], dim=-1)



# ── Glycolysis scaffolds (full 22-state oracle + 12/8/4-state reductions) ──
# Centralised here; the data-generation accessor lives in sim/glycolysis.py.

class GlycolysisOracle22Scaffold(MechanisticScaffold):
    """
    Rich non-oscillatory glycolysis oracle.

    This is a synthetic controlled benchmark inspired by EMP glycolysis. It is not a
    calibrated published yeast glycolysis model. The scaffold keeps many pathway
    intermediates, cofactors, a lactate branch, and hidden inhibitor pools.

    States (22):
      0  Glc       glucose
      1  G6P       glucose-6-phosphate
      2  F6P       fructose-6-phosphate
      3  FBP       fructose-1,6-bisphosphate
      4  GAP       glyceraldehyde-3-phosphate
      5  DHAP      dihydroxyacetone phosphate
      6  BPG13     1,3-bisphosphoglycerate
      7  PG3       3-phosphoglycerate
      8  PG2       2-phosphoglycerate
      9  PEP       phosphoenolpyruvate
      10 Pyr       pyruvate
      11 Lac       lactate
      12 ATP
      13 ADP
      14 NAD
      15 NADH
      16 I_HK      inhibitor pool for Glc -> G6P
      17 I_PFK     inhibitor pool for F6P -> FBP
      18 I_GAPDH   inhibitor pool for GAP -> BPG13
      19 I_PGK     inhibitor pool for BPG13 -> PG3
      20 I_ENO     inhibitor pool for PG2 -> PEP
      21 I_PK      inhibitor pool for PEP -> Pyr

    Inputs / boluses:
      glucose_bolus -> Glc
      atp_bolus     -> ATP
      adp_bolus     -> ADP
      nad_bolus     -> NAD
      nadh_bolus    -> NADH

      hk_inhib_bolus    -> I_HK
      pfk_inhib_bolus   -> I_PFK
      gapdh_inhib_bolus -> I_GAPDH
      pgk_inhib_bolus   -> I_PGK
      eno_inhib_bolus   -> I_ENO
      pk_inhib_bolus    -> I_PK

    Primary observed species:
      Glc, Pyr, ATP, NADH

    Primary observed indices:
      [0, 10, 12, 15]

    Parameters theta (33):
      0  k_hk
      1  k_pgi_f
      2  k_pgi_r
      3  k_pfk
      4  k_ald_f
      5  k_ald_r
      6  k_tpi_f
      7  k_tpi_r
      8  k_gapdh
      9  k_pgk
      10 k_pgm_f
      11 k_pgm_r
      12 k_eno
      13 k_pk
      14 k_ldh
      15 k_pyr_sink
      16 k_atpase
      17 k_nadh_ox
      18 k_leak_hex
      19 k_leak_tri
      20 k_leak_lower
      21 K_I_HK
      22 K_I_PFK
      23 K_I_GAPDH
      24 K_I_PGK
      25 K_I_ENO
      26 K_I_PK
      27 k_I_HK_decay
      28 k_I_PFK_decay
      29 k_I_GAPDH_decay
      30 k_I_PGK_decay
      31 k_I_ENO_decay
      32 k_I_PK_decay

    Inhibition model:
      activity_j = K_I_j / (K_I_j + I_j)

    This is deliberately mass-action / coarse-grained effective kinetics, not
    Michaelis--Menten kinetics. This keeps the oracle relaxation-like and avoids
    feedback-driven oscillatory behavior.
    """

    def __init__(self):
        super().__init__(P=22, theta_dim=33)
        self.state_names = [
            "Glc", "G6P", "F6P", "FBP", "GAP", "DHAP",
            "BPG13", "PG3", "PG2", "PEP", "Pyr", "Lac",
            "ATP", "ADP", "NAD", "NADH",
            "I_HK", "I_PFK", "I_GAPDH", "I_PGK", "I_ENO", "I_PK",
        ]

        self.theta_lo_vec = [
            1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5,
            1e-4, 1e-5, 1e-4, 1e-4, 1e-4, 1e-5,
            1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5,
            1e-6, 1e-6, 1e-6,
            1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4,
            1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5,
        ]

        self.theta_hi_vec = [
            50.0, 50.0, 50.0, 50.0, 50.0, 20.0,
            50.0, 20.0, 50.0, 50.0, 50.0, 20.0,
            50.0, 50.0, 20.0, 10.0, 10.0, 10.0,
            5.0, 5.0, 5.0,
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
        ]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        (
            Glc, G6P, F6P, FBP, GAP, DHAP,
            BPG13, PG3, PG2, PEP, Pyr, Lac,
            ATP, ADP, NAD, NADH,
            I_HK, I_PFK, I_GAPDH, I_PGK, I_ENO, I_PK,
        ) = y.unbind(dim=-1)

        (
            k_hk, k_pgi_f, k_pgi_r, k_pfk, k_ald_f, k_ald_r,
            k_tpi_f, k_tpi_r, k_gapdh, k_pgk, k_pgm_f, k_pgm_r,
            k_eno, k_pk, k_ldh, k_pyr_sink, k_atpase, k_nadh_ox,
            k_leak_hex, k_leak_tri, k_leak_lower,
            K_I_HK, K_I_PFK, K_I_GAPDH, K_I_PGK, K_I_ENO, K_I_PK,
            k_I_HK_decay, k_I_PFK_decay, k_I_GAPDH_decay,
            k_I_PGK_decay, k_I_ENO_decay, k_I_PK_decay,
        ) = theta.unbind(dim=-1)

        eps = 1e-8

        # Nonnegative states.
        Glc_p = torch.clamp_min(Glc, 0.0)
        G6P_p = torch.clamp_min(G6P, 0.0)
        F6P_p = torch.clamp_min(F6P, 0.0)
        FBP_p = torch.clamp_min(FBP, 0.0)
        GAP_p = torch.clamp_min(GAP, 0.0)
        DHAP_p = torch.clamp_min(DHAP, 0.0)
        BPG13_p = torch.clamp_min(BPG13, 0.0)
        PG3_p = torch.clamp_min(PG3, 0.0)
        PG2_p = torch.clamp_min(PG2, 0.0)
        PEP_p = torch.clamp_min(PEP, 0.0)
        Pyr_p = torch.clamp_min(Pyr, 0.0)
        Lac_p = torch.clamp_min(Lac, 0.0)
        ATP_p = torch.clamp_min(ATP, 0.0)
        ADP_p = torch.clamp_min(ADP, 0.0)
        NAD_p = torch.clamp_min(NAD, 0.0)
        NADH_p = torch.clamp_min(NADH, 0.0)

        I_HK_p = torch.clamp_min(I_HK, 0.0)
        I_PFK_p = torch.clamp_min(I_PFK, 0.0)
        I_GAPDH_p = torch.clamp_min(I_GAPDH, 0.0)
        I_PGK_p = torch.clamp_min(I_PGK, 0.0)
        I_ENO_p = torch.clamp_min(I_ENO, 0.0)
        I_PK_p = torch.clamp_min(I_PK, 0.0)

        # Nonnegative parameters.
        params_nonnegative = [
            k_hk, k_pgi_f, k_pgi_r, k_pfk, k_ald_f, k_ald_r,
            k_tpi_f, k_tpi_r, k_gapdh, k_pgk, k_pgm_f, k_pgm_r,
            k_eno, k_pk, k_ldh, k_pyr_sink, k_atpase, k_nadh_ox,
            k_leak_hex, k_leak_tri, k_leak_lower,
            K_I_HK, K_I_PFK, K_I_GAPDH, K_I_PGK, K_I_ENO, K_I_PK,
            k_I_HK_decay, k_I_PFK_decay, k_I_GAPDH_decay,
            k_I_PGK_decay, k_I_ENO_decay, k_I_PK_decay,
        ]
        (
            k_hk, k_pgi_f, k_pgi_r, k_pfk, k_ald_f, k_ald_r,
            k_tpi_f, k_tpi_r, k_gapdh, k_pgk, k_pgm_f, k_pgm_r,
            k_eno, k_pk, k_ldh, k_pyr_sink, k_atpase, k_nadh_ox,
            k_leak_hex, k_leak_tri, k_leak_lower,
            K_I_HK, K_I_PFK, K_I_GAPDH, K_I_PGK, K_I_ENO, K_I_PK,
            k_I_HK_decay, k_I_PFK_decay, k_I_GAPDH_decay,
            k_I_PGK_decay, k_I_ENO_decay, k_I_PK_decay,
        ) = [torch.clamp_min(p, eps) for p in params_nonnegative]

        # Inhibitor-dependent activities.
        a_hk = K_I_HK / (K_I_HK + I_HK_p + eps)
        a_pfk = K_I_PFK / (K_I_PFK + I_PFK_p + eps)
        a_gapdh = K_I_GAPDH / (K_I_GAPDH + I_GAPDH_p + eps)
        a_pgk = K_I_PGK / (K_I_PGK + I_PGK_p + eps)
        a_eno = K_I_ENO / (K_I_ENO + I_ENO_p + eps)
        a_pk = K_I_PK / (K_I_PK + I_PK_p + eps)

        # Glycolytic reaction rates.
        r_hk = a_hk * k_hk * Glc_p * ATP_p
        r_pgi = k_pgi_f * G6P_p - k_pgi_r * F6P_p
        r_pfk = a_pfk * k_pfk * F6P_p * ATP_p
        r_ald = k_ald_f * FBP_p - k_ald_r * GAP_p * DHAP_p
        r_tpi = k_tpi_f * DHAP_p - k_tpi_r * GAP_p
        r_gapdh = a_gapdh * k_gapdh * GAP_p * NAD_p
        r_pgk = a_pgk * k_pgk * BPG13_p * ADP_p
        r_pgm = k_pgm_f * PG3_p - k_pgm_r * PG2_p
        r_eno = a_eno * k_eno * PG2_p
        r_pk = a_pk * k_pk * PEP_p * ADP_p

        # Branches and dissipative terms.
        r_ldh = k_ldh * Pyr_p * NADH_p
        r_pyr_sink = k_pyr_sink * Pyr_p
        r_atpase = k_atpase * ATP_p
        r_nadh_ox = k_nadh_ox * NADH_p

        l_G6P = k_leak_hex * G6P_p
        l_F6P = k_leak_hex * F6P_p
        l_FBP = k_leak_hex * FBP_p
        l_GAP = k_leak_tri * GAP_p
        l_DHAP = k_leak_tri * DHAP_p
        l_BPG13 = k_leak_lower * BPG13_p
        l_PG3 = k_leak_lower * PG3_p
        l_PG2 = k_leak_lower * PG2_p
        l_PEP = k_leak_lower * PEP_p

        # State equations.
        dGlc = -r_hk
        dG6P = r_hk - r_pgi - l_G6P
        dF6P = r_pgi - r_pfk - l_F6P
        dFBP = r_pfk - r_ald - l_FBP
        dGAP = r_ald + r_tpi - r_gapdh - l_GAP
        dDHAP = r_ald - r_tpi - l_DHAP
        dBPG13 = r_gapdh - r_pgk - l_BPG13
        dPG3 = r_pgk - r_pgm - l_PG3
        dPG2 = r_pgm - r_eno - l_PG2
        dPEP = r_eno - r_pk - l_PEP
        dPyr = r_pk - r_ldh - r_pyr_sink
        dLac = r_ldh

        dATP = -r_hk - r_pfk + r_pgk + r_pk - r_atpase
        dADP = r_hk + r_pfk - r_pgk - r_pk + r_atpase
        dNAD = -r_gapdh + r_ldh + r_nadh_ox
        dNADH = r_gapdh - r_ldh - r_nadh_ox

        dI_HK = -k_I_HK_decay * I_HK_p
        dI_PFK = -k_I_PFK_decay * I_PFK_p
        dI_GAPDH = -k_I_GAPDH_decay * I_GAPDH_p
        dI_PGK = -k_I_PGK_decay * I_PGK_p
        dI_ENO = -k_I_ENO_decay * I_ENO_p
        dI_PK = -k_I_PK_decay * I_PK_p

        return torch.stack(
            (
                dGlc, dG6P, dF6P, dFBP, dGAP, dDHAP,
                dBPG13, dPG3, dPG2, dPEP, dPyr, dLac,
                dATP, dADP, dNAD, dNADH,
                dI_HK, dI_PFK, dI_GAPDH, dI_PGK, dI_ENO, dI_PK,
            ),
            dim=-1,
        )


# =============================================================================
# 2) REDUCED 12-state scaffold
# =============================================================================

class GlycolysisReduced12Scaffold(MechanisticScaffold):
    """
    Reduced glycolysis scaffold with coarse pathway pools.

    States (12):
      0  Glc
      1  HexP   coarse G6P/F6P/FBP pool
      2  TriP   coarse GAP/DHAP pool
      3  BPG13
      4  PG3
      5  PEP
      6  Pyr
      7  Lac
      8  ATP
      9  ADP
      10 NAD
      11 NADH

    Observed species:
      Glc, Pyr, ATP, NADH

    Observed indices:
      [0, 6, 8, 11]

    Inhibitor boluses are not states in this reduced model. They should be passed to the
    encoder input history so that theta(t) can absorb their hidden effect.
    """

    def __init__(self):
        super().__init__(P=12, theta_dim=12)
        self.state_names = [
            "Glc", "HexP", "TriP", "BPG13", "PG3", "PEP",
            "Pyr", "Lac", "ATP", "ADP", "NAD", "NADH",
        ]
        self.theta_lo_vec = [1e-5] * 12
        self.theta_hi_vec = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 20.0, 10.0, 10.0, 10.0, 5.0, 5.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        Glc, HexP, TriP, BPG13, PG3, PEP, Pyr, Lac, ATP, ADP, NAD, NADH = y.unbind(dim=-1)

        (
            k_hk_eff,
            k_commit_eff,
            k_gapdh_eff,
            k_pgk_eff,
            k_lower_eff,
            k_pk_eff,
            k_ldh_eff,
            k_pyr_sink,
            k_atpase,
            k_nadh_ox,
            k_hex_leak,
            k_tri_leak,
        ) = theta.unbind(dim=-1)

        Glc_p = torch.clamp_min(Glc, 0.0)
        HexP_p = torch.clamp_min(HexP, 0.0)
        TriP_p = torch.clamp_min(TriP, 0.0)
        BPG13_p = torch.clamp_min(BPG13, 0.0)
        PG3_p = torch.clamp_min(PG3, 0.0)
        PEP_p = torch.clamp_min(PEP, 0.0)
        Pyr_p = torch.clamp_min(Pyr, 0.0)
        ATP_p = torch.clamp_min(ATP, 0.0)
        ADP_p = torch.clamp_min(ADP, 0.0)
        NAD_p = torch.clamp_min(NAD, 0.0)
        NADH_p = torch.clamp_min(NADH, 0.0)

        (
            k_hk_eff,
            k_commit_eff,
            k_gapdh_eff,
            k_pgk_eff,
            k_lower_eff,
            k_pk_eff,
            k_ldh_eff,
            k_pyr_sink,
            k_atpase,
            k_nadh_ox,
            k_hex_leak,
            k_tri_leak,
        ) = [torch.clamp_min(p, 0.0) for p in (
            k_hk_eff, k_commit_eff, k_gapdh_eff, k_pgk_eff,
            k_lower_eff, k_pk_eff, k_ldh_eff, k_pyr_sink,
            k_atpase, k_nadh_ox, k_hex_leak, k_tri_leak,
        )]

        r_hk = k_hk_eff * Glc_p * ATP_p
        r_commit = k_commit_eff * HexP_p * ATP_p
        r_gapdh = k_gapdh_eff * TriP_p * NAD_p
        r_pgk = k_pgk_eff * BPG13_p * ADP_p
        r_lower = k_lower_eff * PG3_p
        r_pk = k_pk_eff * PEP_p * ADP_p
        r_ldh = k_ldh_eff * Pyr_p * NADH_p

        r_pyr_sink = k_pyr_sink * Pyr_p
        r_atpase = k_atpase * ATP_p
        r_nadh_ox = k_nadh_ox * NADH_p
        r_hex_leak = k_hex_leak * HexP_p
        r_tri_leak = k_tri_leak * TriP_p

        dGlc = -r_hk
        dHexP = r_hk - r_commit - r_hex_leak
        dTriP = 2.0 * r_commit - r_gapdh - r_tri_leak
        dBPG13 = r_gapdh - r_pgk
        dPG3 = r_pgk - r_lower
        dPEP = r_lower - r_pk
        dPyr = r_pk - r_ldh - r_pyr_sink
        dLac = r_ldh

        dATP = -r_hk - r_commit + r_pgk + r_pk - r_atpase
        dADP = r_hk + r_commit - r_pgk - r_pk + r_atpase
        dNAD = -r_gapdh + r_ldh + r_nadh_ox
        dNADH = r_gapdh - r_ldh - r_nadh_ox

        return torch.stack(
            (dGlc, dHexP, dTriP, dBPG13, dPG3, dPEP, dPyr, dLac, dATP, dADP, dNAD, dNADH),
            dim=-1,
        )


# =============================================================================
# 3) REDUCED 8-state scaffold
# =============================================================================

class GlycolysisReduced8Scaffold(MechanisticScaffold):
    """
    More compressed glycolysis scaffold.

    States (8):
      0 Glc
      1 SugarP   coarse phosphorylated sugar/triose pool
      2 PEP
      3 Pyr
      4 Lac
      5 ATP
      6 NAD
      7 NADH

    Observed species:
      Glc, Pyr, ATP, NADH

    Observed indices:
      [0, 3, 5, 7]
    """

    def __init__(self):
        super().__init__(P=8, theta_dim=9)
        self.state_names = ["Glc", "SugarP", "PEP", "Pyr", "Lac", "ATP", "NAD", "NADH"]
        self.theta_lo_vec = [1e-5] * 9
        self.theta_hi_vec = [50.0, 50.0, 50.0, 20.0, 10.0, 10.0, 10.0, 5.0, 5.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        Glc, SugarP, PEP, Pyr, Lac, ATP, NAD, NADH = y.unbind(dim=-1)

        (
            k_hk_eff,
            k_path_eff,
            k_pk_eff,
            k_ldh_eff,
            k_pyr_sink,
            k_atpase,
            k_nadh_ox,
            k_sugar_leak,
            k_pep_leak,
        ) = theta.unbind(dim=-1)

        Glc_p = torch.clamp_min(Glc, 0.0)
        SugarP_p = torch.clamp_min(SugarP, 0.0)
        PEP_p = torch.clamp_min(PEP, 0.0)
        Pyr_p = torch.clamp_min(Pyr, 0.0)
        ATP_p = torch.clamp_min(ATP, 0.0)
        NAD_p = torch.clamp_min(NAD, 0.0)
        NADH_p = torch.clamp_min(NADH, 0.0)

        (
            k_hk_eff,
            k_path_eff,
            k_pk_eff,
            k_ldh_eff,
            k_pyr_sink,
            k_atpase,
            k_nadh_ox,
            k_sugar_leak,
            k_pep_leak,
        ) = [torch.clamp_min(p, 0.0) for p in (
            k_hk_eff, k_path_eff, k_pk_eff, k_ldh_eff,
            k_pyr_sink, k_atpase, k_nadh_ox, k_sugar_leak, k_pep_leak,
        )]

        r_hk = k_hk_eff * Glc_p * ATP_p
        r_path = k_path_eff * SugarP_p * NAD_p
        r_pk = k_pk_eff * PEP_p
        r_ldh = k_ldh_eff * Pyr_p * NADH_p

        r_pyr_sink = k_pyr_sink * Pyr_p
        r_atpase = k_atpase * ATP_p
        r_nadh_ox = k_nadh_ox * NADH_p
        r_sugar_leak = k_sugar_leak * SugarP_p
        r_pep_leak = k_pep_leak * PEP_p

        dGlc = -r_hk
        dSugarP = r_hk - r_path - r_sugar_leak
        dPEP = r_path - r_pk - r_pep_leak
        dPyr = r_pk - r_ldh - r_pyr_sink
        dLac = r_ldh

        dATP = -r_hk + r_pk - r_atpase
        dNAD = -r_path + r_ldh + r_nadh_ox
        dNADH = r_path - r_ldh - r_nadh_ox

        return torch.stack((dGlc, dSugarP, dPEP, dPyr, dLac, dATP, dNAD, dNADH), dim=-1)


# =============================================================================
# 4) REDUCED 4-state scaffold
# =============================================================================

class GlycolysisReduced4Scaffold(MechanisticScaffold):
    """
    Observed-state-only reduced glycolysis scaffold.

    States (4):
      0 Glc
      1 Pyr
      2 ATP
      3 NADH

    Observed species:
      Glc, Pyr, ATP, NADH

    Observed indices:
      [0, 1, 2, 3]

    All pathway intermediates, cofactors other than ATP/NADH, lactate, and inhibitor pools
    are omitted. Inhibitor additions should be passed only to the causal encoder.
    """

    def __init__(self):
        super().__init__(P=4, theta_dim=7)
        self.state_names = ["Glc", "Pyr", "ATP", "NADH"]

        # All bounds positive: ATP and NADH are net *produced* from glucose
        # breakdown (2 ATP, 2 NADH per glucose), so k_atp_eff and k_nadh_eff
        # are sign-fixed positive. (Earlier signed bounds broke `log_gamma`,
        # which requires lo and hi to share sign.)
        self.theta_lo_vec = [
            1e-5,   # k_glc_use
            1e-5,   # k_pyr_prod
            1e-5,   # k_pyr_sink
            1e-5,   # k_atp_eff
            1e-5,   # k_atp_loss
            1e-5,   # k_nadh_eff
            1e-5,   # k_nadh_loss
        ]

        self.theta_hi_vec = [
            50.0,  # k_glc_use
            50.0,  # k_pyr_prod
            20.0,  # k_pyr_sink
            10.0,  # k_atp_eff
            10.0,  # k_atp_loss
            10.0,  # k_nadh_eff
            10.0,  # k_nadh_loss
        ]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        Glc, Pyr, ATP, NADH = y.unbind(dim=-1)

        (
            k_glc_use,
            k_pyr_prod,
            k_pyr_sink,
            k_atp_eff,
            k_atp_loss,
            k_nadh_eff,
            k_nadh_loss,
        ) = theta.unbind(dim=-1)

        Glc_p = torch.clamp_min(Glc, 0.0)
        Pyr_p = torch.clamp_min(Pyr, 0.0)
        ATP_p = torch.clamp_min(ATP, 0.0)
        NADH_p = torch.clamp_min(NADH, 0.0)

        k_glc_use = torch.clamp_min(k_glc_use, 0.0)
        k_pyr_prod = torch.clamp_min(k_pyr_prod, 0.0)
        k_pyr_sink = torch.clamp_min(k_pyr_sink, 0.0)
        k_atp_loss = torch.clamp_min(k_atp_loss, 0.0)
        k_nadh_loss = torch.clamp_min(k_nadh_loss, 0.0)

        r_glc = k_glc_use * Glc_p
        r_pyr = k_pyr_prod * Glc_p
        r_sink = k_pyr_sink * Pyr_p

        dGlc = -r_glc
        dPyr = r_pyr - r_sink

        # Signed effective closure terms.
        dATP = k_atp_eff * Glc_p - k_atp_loss * ATP_p
        dNADH = k_nadh_eff * Glc_p - k_nadh_loss * NADH_p

        return torch.stack((dGlc, dPyr, dATP, dNADH), dim=-1)


# =============================================================================
# Registry and metadata
# =============================================================================


class TXTLResourceandMaturationDNAScaffold(MechanisticScaffold):
    """
    6-state TXTL scaffold with DNA as an explicit, bolus-driven state.

    The mechanism is `TXTL_mRNAMaturation`, with DNA promoted
    from an exogenous scalar to a latent state so no scaffold-API change is
    needed: the dataset's `u_to_y_jump` routes the "DNA c" (dilution-corrected
    concentration delta) column of u_seq onto state idx 5, and dDNA/dt = 0
    between jumps — so y[..., 5] at step k is exactly cumsum("DNA c") up to k.

    States (6): R (resource pool), m (immature mRNA), mm (mature mRNA,
                observed as Broccoli), p (immature protein),
                pm (mature protein, observed as mCherry / 2), DNA

    Parameters θ (6):
      0  lam    : resource decay rate
      1  VTXmax : transcription rate (per DNA per R)
      2  kdm    : mRNA degradation rate (applies to both m and mm)
      3  VTLmax : translation rate (per total mRNA per R)
      4  kmt    : protein maturation rate (p → pm)
      5  kmatm  : mRNA maturation rate (m → mm)

    Observed indices within P: [2, 4]  (mm=Broccoli, pm=mCherry/2)
    Use with: datasets/real_ivtt_full.npz (layout='full')
    """
    def __init__(self):
        super().__init__(P=7, theta_dim=7)
        self.state_names = ["R", "O", "m", "mm", "p", "pm", "DNA"]
        # log-uniform bounds for TXTL_mRNAMaturation
        self.theta_lo_vec = [1e-6, 1e-6, 3e-5, 1e-5, 3e-5, 1e-5, 5e-5]
        self.theta_hi_vec = [5e-4, 5e-4, 1.2e-1, 1e-2, 8e-2, 3.5e-4, 3.5e-3]
    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        R, O, m, mm, p, pm, DNA = y.unbind(dim=-1)
        lam,lam_O, VTXmax, kdm, VTLmax, kmt, kmatm = theta.unbind(dim=-1)

        R_p   = torch.clamp_min(R,   0.0)
        O_p   = torch.clamp_min(O,   0.0)
        m_p   = torch.clamp_min(m,   0.0)
        mm_p  = torch.clamp_min(mm,  0.0)
        p_p   = torch.clamp_min(p,   0.0)
        DNA_p = torch.clamp_min(DNA, 0.0)

        dR   = -lam * R_p
        dO   = -lam_O * O_p
        dm   = R_p * VTXmax * DNA_p - (kdm + kmatm) * m_p
        dmm  = kmatm * m_p - kdm * mm_p
        dp   = R_p * VTLmax * (m_p + mm_p) - kmt * p_p
        dpm  = O_p * kmt * p_p
        dDNA = torch.zeros_like(DNA)

        return torch.stack((dR, dO, dm, dmm, dp, dpm, dDNA), dim=-1)
class TXTLModel3_TwoStateScaffold(MechanisticScaffold):
    """
    Model 3 (two-state observable ODE) from new_scaffolds.tex.

        dM/dt = v_TX * DNA - k_M * M
        dP/dt = v_TL * M   - k_P * P

    The tex writes v_TX(t) and v_TL(t) as time-varying drives produced by the
    encoder, so they live in theta. DNA is kept as a bolus-driven tracker
    state so existing data wiring works unchanged.

    States (3): M, P, DNA
    theta (4): v_TX, v_TL, k_M, k_P
    Observed indices: [0, 1]   (M = mRNA, P = protein)
    """
    def __init__(self):
        super().__init__(P=3, theta_dim=4)
        self.state_names = ["M", "P", "DNA"]
        # Same order-of-magnitude box as TXTLMaturationDNAScaffold.
        self.theta_lo_vec = [3e-5, 3e-5, 1e-5, 1e-5]
        self.theta_hi_vec = [1.2e-1, 8e-2, 1e-2, 1e-2]
        # Dataset y_seq carries (mm, pm) at dataset cols 3, 5; in this scaffold
        # they correspond to M (idx 0) and P (idx 1).
        self.obs_state_idx = [0, 1]
        self.control_state_map = {"DNA c": 2}

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        M, P, DNA = y.unbind(dim=-1)
        v_TX, v_TL, k_M, k_P = theta.unbind(dim=-1)

        M_p   = torch.clamp_min(M,   0.0)
        P_p   = torch.clamp_min(P,   0.0)
        DNA_p = torch.clamp_min(DNA, 0.0)

        dM   = v_TX * DNA_p - k_M * M_p
        dP   = v_TL * M_p   - k_P * P_p
        dDNA = torch.zeros_like(DNA)

        return torch.stack((dM, dP, dDNA), dim=-1)
class TXTLModel4_ThreeStateScaffold(MechanisticScaffold):
    """
    Model 4 (three-state TX-TL-maturation ODE) from new_scaffolds.tex.

        dM/dt        = v_TX * DNA - k_M * M
        dP_imm/dt    = v_TL * M   - k_mat * P_imm - k_degp * P_imm
        dP_fluor/dt  = k_mat * P_imm

    States (4): M, P_imm, P_fluor, DNA
    theta (5): v_TX, v_TL, k_M, k_mat, k_degp
    Observed indices: [0, 2]   (M, P_fluor)
    """
    def __init__(self):
        super().__init__(P=4, theta_dim=5)
        self.state_names = ["M", "P_imm", "P_fluor", "DNA"]
        self.theta_lo_vec = [3e-5, 3e-5, 1e-5, 1e-5, 1e-7]
        self.theta_hi_vec = [1.2e-1, 8e-2, 1e-2, 3.5e-4, 1e-3]
        # (mm, pm) -> (M, P_fluor)
        self.obs_state_idx = [0, 2]
        self.control_state_map = {"DNA c": 3}

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        M, P_imm, P_fluor, DNA = y.unbind(dim=-1)
        v_TX, v_TL, k_M, k_mat, k_degp = theta.unbind(dim=-1)

        M_p     = torch.clamp_min(M,     0.0)
        P_imm_p = torch.clamp_min(P_imm, 0.0)
        DNA_p   = torch.clamp_min(DNA,   0.0)

        dM       = v_TX * DNA_p - k_M * M_p
        dP_imm   = v_TL * M_p   - (k_mat + k_degp) * P_imm_p
        dP_fluor = k_mat * P_imm_p
        dDNA     = torch.zeros_like(DNA)

        return torch.stack((dM, dP_imm, dP_fluor, dDNA), dim=-1)
class TXTLModel7_BgFixed(MechanisticScaffold):
    """
    M7 with all fixes from TXTLModel7_FullFixed PLUS learnable per-reagent
    background concentrations that prevent zero-gate collapse on real data.

    Root cause: g_expr = g_T7 * g_NTP * g_AA * g_Mg * g_K collapses to 0
    whenever any reagent bolus is 0.  For ~16% of real samples K-Glut=0,
    yet those samples still express because lysate provides background K.
    No adjustment of K_K (half-saturation) can fix this; g_K = K_p/(K_K+K_p)
    is 0 when K_p=0 regardless of K_K.

    Fix: add learned background offsets (T7_bg, NTP_bg, AA_bg, Mg_bg, K_bg)
    to each gate.  The GRU encoder predicts sample-specific backgrounds,
    effectively learning "this K-Glut=0 sample still has background K from
    lysate."  Synth kill detection is unaffected: zero DNA kills production
    via the R*V_TX*DNA term regardless of g_expr.

    theta (17): [lam, lam_O, V_TX, k_dm, V_TL, kmt, kmatm,
                  K_T7, K_NTP, K_AA, K_Mg, K_K,
                  T7_bg, NTP_bg, AA_bg, Mg_bg, K_bg]
    """
    def __init__(self):
        super().__init__(P=12, theta_dim=17)
        self.state_names = [
            "R", "O", "m", "mm", "p", "pm",
            "T7", "NTP", "AA", "Mg", "K_ion", "DNA",
        ]
        self.theta_lo_vec = [
            1e-6, 1e-6, 3e-5, 1e-5, 3e-5, 1e-5, 5e-5,   # kinetics (same as FullFixed)
            1.0,   5.0,  10.0, 0.5,  1.0,                 # K_T7..K_K (nL scale)
            0.1,   0.1,  0.1,  0.1,  0.1,                 # T7_bg, NTP_bg, AA_bg, Mg_bg, K_bg
        ]
        self.theta_hi_vec = [
            5e-4, 5e-4, 1.2e-1, 1e-2, 8e-2, 1.0e-3, 3.5e-3,
            5000., 15000., 20000., 2000., 8000.,
            200., 2000., 2000., 500., 2000.,               # background upper bounds (nL scale)
        ]
        self.obs_state_idx = [3, 5]
        self.control_state_map = {
            "DNA c": 11, "T7RNAP": 6, "NTPs": 7, "AA": 8,
            "Mg-Glut": 9, "K-Glut": 10,
            "Lysate 2%PEG": 8,
            "FB": 7,
        }

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        R, O, m, mm, p, pm, T7, NTP, AA, Mg, K_ion, DNA = y.unbind(dim=-1)
        (lam, lam_O, V_TX, kdm, V_TL, kmt, kmatm,
         K_T7, K_NTP, K_AA, K_Mg, K_K,
         T7_bg, NTP_bg, AA_bg, Mg_bg, K_bg) = theta.unbind(dim=-1)

        R_p   = torch.clamp_min(R,   0.0)
        O_p   = torch.clamp_min(O,   0.0)
        m_p   = torch.clamp_min(m,   0.0)
        mm_p  = torch.clamp_min(mm,  0.0)
        p_p   = torch.clamp_min(p,   0.0)
        DNA_p = torch.clamp_min(DNA, 0.0)
        T7_eff  = torch.clamp_min(T7,     0.0) + T7_bg
        NTP_eff = torch.clamp_min(NTP,    0.0) + NTP_bg
        AA_eff  = torch.clamp_min(AA,     0.0) + AA_bg
        Mg_eff  = torch.clamp_min(Mg,     0.0) + Mg_bg
        K_eff   = torch.clamp_min(K_ion,  0.0) + K_bg

        eps = 1e-12
        g_T7  = T7_eff  / (K_T7  + T7_eff  + eps)
        g_NTP = NTP_eff / (K_NTP + NTP_eff + eps)
        g_AA  = AA_eff  / (K_AA  + AA_eff  + eps)
        g_Mg  = Mg_eff  / (K_Mg  + Mg_eff  + eps)
        g_K   = K_eff   / (K_K   + K_eff   + eps)
        g_expr = g_T7 * g_NTP * g_AA * g_Mg * g_K

        dR   = -lam * R_p
        dO   = -lam_O * O_p
        dm   = g_expr * R_p * V_TX * DNA_p - (kdm + kmatm) * m_p
        dmm  = kmatm * m_p - kdm * mm_p
        dp   = g_expr * R_p * V_TL * (m_p + mm_p) - kmt * p_p
        dpm  = kmt * p_p

        zero = torch.zeros_like(DNA)
        return torch.stack(
            (dR, dO, dm, dmm, dp, dpm, zero, zero, zero, zero, zero, zero),
            dim=-1,
        )
class TXTLModel8_BgFixed(MechanisticScaffold):
    """
    M8 with all fixes from TXTLModel8_FullFixed PLUS learnable per-resource
    background concentrations that prevent zero-gate collapse on real data.

    Same root cause as M7: K_ion=0 (16% real samples, no K-Glut bolus) →
    f_K=0 → v_TX=v_TL=0 → predicted mm=pm=0 despite actual expression.

    Fix: add per-resource background offsets (E_bg, A_bg, Mg_bg, K_bg, C_bg)
    to the Michaelis gates.  The GRU encoder learns sample-specific offsets,
    effectively learning background resource availability from lysate.

    Note: T_p (T7 polymerase) enters v_TX as a direct multiplier, not a gate,
    so T_bg is not needed here — it would require a structural change.  The
    dominant zero-gate issue in real data is K and potentially C (PEG crowding,
    mostly addressed already by Fix 4 / Lysate→[A,C] mapping).

    theta (21): [alpha_TX, alpha_TL, k_E, k_A, k_T, k_W,
                  K_E, K_A, K_Mg, K_K, K_C, K_W,
                  kdm, kmatm, kmt, beta_W,
                  E_bg, A_bg, Mg_bg, K_bg, C_bg]
    """
    def __init__(self):
        super().__init__(P=12, theta_dim=21)
        self.state_names = [
            "E", "A", "T", "Mg", "K_ion", "C", "W",
            "m", "mm", "p", "pm", "DNA",
        ]
        self.theta_lo_vec = [
            3e-5, 3e-5,                         # alpha_TX, alpha_TL
            1e-7, 1e-7, 1e-7, 1e-7,             # k_E, k_A, k_T, k_W
            5.0,  10.0, 0.5,  1.0, 5.0, 1e-4,   # K_E, K_A, K_Mg, K_K, K_C (nL), K_W
            1e-5, 5e-5, 1e-5,                   # k_dm, k_matm, k_mt
            1e-6,                               # beta_W
            0.1, 0.1, 0.1, 0.1, 0.1,            # E_bg, A_bg, Mg_bg, K_bg, C_bg
        ]
        self.theta_hi_vec = [
            1.2e-1, 8e-2,
            1e-3, 1e-3, 1e-3, 1e-3,
            15000., 20000., 2000., 8000., 8000., 100.,
            1e-2, 3.5e-3, 3.5e-4,
            1e-2,
            2000., 2000., 500., 2000., 500.,     # background upper bounds (nL scale)
        ]
        self.obs_state_idx = [8, 10]
        self.control_state_map = {
            "DNA c": 11, "NTPs": 0, "FB": 0, "Maltose": 0,
            "AA": 1, "T7RNAP": 2, "Mg-Glut": 3, "K-Glut": 4, "PEG8000": 5,
            "Lysate 2%PEG": [1, 5],
        }

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        E, A, T, Mg, K_ion, C, W, m, mm, p, pm, DNA = y.unbind(dim=-1)
        (alpha_TX, alpha_TL,
         k_E, k_A, k_T, k_W,
         K_E, K_A, K_Mg, K_K, K_C, K_W,
         kdm, kmatm, kmt, beta_W,
         E_bg, A_bg, Mg_bg, K_bg, C_bg) = theta.unbind(dim=-1)

        T_p   = torch.clamp_min(T,   0.0)
        W_p   = torch.clamp_min(W,   0.0)
        m_p   = torch.clamp_min(m,   0.0)
        mm_p  = torch.clamp_min(mm,  0.0)
        p_p   = torch.clamp_min(p,   0.0)
        DNA_p = torch.clamp_min(DNA, 0.0)
        E_eff  = torch.clamp_min(E,     0.0) + E_bg
        A_eff  = torch.clamp_min(A,     0.0) + A_bg
        Mg_eff = torch.clamp_min(Mg,    0.0) + Mg_bg
        K_eff  = torch.clamp_min(K_ion, 0.0) + K_bg
        C_eff  = torch.clamp_min(C,     0.0) + C_bg

        eps = 1e-12
        f_E  = E_eff  / (K_E  + E_eff  + eps)
        f_A  = A_eff  / (K_A  + A_eff  + eps)
        f_Mg = Mg_eff / (K_Mg + Mg_eff + eps)
        f_K  = K_eff  / (K_K  + K_eff  + eps)
        f_C  = C_eff  / (K_C  + C_eff  + eps)
        f_W  = 1.0    / (1.0  + W_p / (K_W + eps))

        v_TX = alpha_TX * DNA_p * T_p * f_E * f_Mg * f_K * f_C * f_W
        v_TL = alpha_TL * (m_p + mm_p) * f_A * f_E * f_Mg * f_K * f_C * f_W

        dE  = -k_E * torch.clamp_min(E, 0.0)
        dA  = -k_A * torch.clamp_min(A, 0.0)
        dT  = -k_T * T_p
        dMg = torch.zeros_like(Mg)
        dK  = torch.zeros_like(K_ion)
        dC  = torch.zeros_like(C)
        dW  = beta_W * (v_TX + v_TL) - k_W * W_p

        dm   = v_TX - (kdm + kmatm) * m_p
        dmm  = kmatm * m_p - kdm * mm_p
        dp   = v_TL - kmt * p_p
        dpm  = kmt * p_p
        dDNA = torch.zeros_like(DNA)

        return torch.stack(
            (dE, dA, dT, dMg, dK, dC, dW, dm, dmm, dp, dpm, dDNA),
            dim=-1,
        )
class TXTLModel9_EventDark(MechanisticScaffold):
    """
    M9 (event-gated dark-protein maturation) — the bug-fixed dark_stable.

    PROBLEM with txtl_model9_dark_stable / dark_m4 (the FINAL runs):
        Those gate the OXYGEN SINK on the opening event (sink_mask = 1 - open_gate)
        but leave the dark->fluor conversion r = soft-sat(k_ox*O2*P_dark) running
        CONTINUOUSLY. With O2 initialised at 1.0 for *every* sample (old and new
        alike — verified in the dataset y0) the conversion never switches: O2 just
        drifts to ~0.55 and freezes, so r is roughly constant throughout. There is
        no term that releases the dark pool AT the opening event, so the model
        cannot reproduce the flat-then-jump pm trajectory the new (tube-opening)
        samples show. Result: predicted pm plateaus far below the post-opening
        jump (~20 vs ~120 on idx702) and M9 underperforms even M5 on new data.

    FIX (this scaffold): make the opening event gate the conversion itself, the
    way v3a (TXTLModel9_O2SourceA) did, but keep dark_stable's explicit dark pool
    ("savings account") so protein synthesised pre-opening is held until release:

        k_mat_eff = k_mat_base + k_open * open_gate
            k_mat_base : baseline maturation, always on -> matures the dark pool of
                         OLD (always-aerobic) samples; the encoder can keep it small
                         for NEW samples (old/new are identifiable from y0: new start
                         with mm,pm > 0).
            k_open     : extra maturation UNLOCKED at the opening event -> converts
                         the accumulated dark pool into fluorescent protein in a
                         burst, reproducing the post-opening jump.

        dPdark/dt  = k_fold*p - r_safe - k_deg_dark*P_dark
        dPfluor/dt = r_safe,   r_safe = soft-sat(k_mat_eff * P_dark)

    The soft-saturation cap (eigenvalue <= 1/tau, tau=10 min) is retained from
    dark_stable for RK4 stability on the coarse grid. The explicit O2 state is
    dropped: it was inert (IC=1 for all samples, no gating role left), so removing
    it only sheds dead weight — oxygen now enters implicitly as the *reason* the
    opening event unlocks maturation.

    States (8): [R, m, mm, p, P_dark, P_fluor, DNA, tube_opened]
    theta (10): [lam, V_TX, V_TL, k_dm, k_matm,
                  k_fold, k_degp, k_deg_dark, k_mat_base, k_open]
    Observed:   [2, 5]   (mm, P_fluor)
    """

    def __init__(self):
        super().__init__(P=8, theta_dim=10)
        # Same RK4-stability time-scale as dark_stable (10 min = 600 s).
        self.TAU_STABLE_SEC: float = 600.0
        self.state_names = [
            "R", "m", "mm", "p", "P_dark", "P_fluor", "DNA", "tube_opened",
        ]
        #                  lam   V_TX   V_TL  k_dm  k_matm k_fold k_degp k_degD
        self.theta_lo_vec = [1e-6, 3e-5, 3e-5, 1e-5, 5e-5, 1e-5, 1e-7, 1e-7,
                             # k_mat_base  k_open
                             1e-6,         1e-4]
        self.theta_hi_vec = [5e-4, 1.2e-1, 8e-2, 1e-2, 3.5e-3, 3.5e-3, 1e-3, 1e-3,
                             # k_mat_base: up to ~0.1/min so old samples can fully
                             # mature; soft-sat still caps the eigenvalue at 0.1/min.
                             1e-1,
                             # k_open: large so the post-opening rate saturates the
                             # cap -> a sharp burst rather than a slow ramp.
                             1e0]
        self.obs_state_idx = [2, 5]   # mm @ 2, P_fluor @ 5
        self.control_state_map = {"DNA c": 6, "u_open": 7}

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        R, m, mm, p, P_dark, P_fluor, DNA, tube_opened = y.unbind(dim=-1)
        (lam, V_TX, V_TL, kdm, kmatm,
         k_fold, k_degp, k_deg_dark, k_mat_base, k_open) = theta.unbind(dim=-1)

        R_p     = torch.clamp_min(R,      0.0)
        m_p     = torch.clamp_min(m,      0.0)
        mm_p    = torch.clamp_min(mm,     0.0)
        p_p     = torch.clamp_min(p,      0.0)
        Pdark_p = torch.clamp(P_dark, 0.0, 1e4)
        DNA_p   = torch.clamp_min(DNA,    0.0)

        # 0 before opening, 1 after (clamped against integration overshoot).
        open_gate = torch.clamp(tube_opened, 0.0, 1.0)

        # Event-gated maturation rate: baseline (matures old samples) + an
        # opening-unlocked boost (releases the dark pool as the post-opening jump).
        k_mat_eff = k_mat_base + k_open * open_gate

        # Soft-saturate the conversion so the eigenvalue k_mat_eff <= 1/tau,
        # keeping RK4 stable on the coarse grid (identical scheme to dark_stable).
        r_bare = k_mat_eff * Pdark_p
        TAU_MIN = self.TAU_STABLE_SEC / 60.0
        max_rate = Pdark_p / TAU_MIN
        r_safe = max_rate * torch.tanh(r_bare / (max_rate + 1e-12))

        dR        = -lam * R_p
        dm        = R_p * V_TX * DNA_p - (kdm + kmatm) * m_p
        dmm       = kmatm * m_p - kdm * mm_p
        dp        = R_p * V_TL * (m_p + mm_p) - k_fold * p_p - k_degp * p_p
        dPdark    = k_fold * p_p - r_safe - k_deg_dark * Pdark_p
        dPfluor   = r_safe
        dDNA      = torch.zeros_like(DNA)
        dtube     = torch.zeros_like(tube_opened)

        return torch.stack(
            (dR, dm, dmm, dp, dPdark, dPfluor, dDNA, dtube),
            dim=-1,
        )


SCAFFOLDS: dict = {
    "txtl_model3_two_state":            TXTLModel3_TwoStateScaffold(),       # M3
    "txtl_model4_three_state":          TXTLModel4_ThreeStateScaffold(),     # M4
    "txtl_resource_and_maturation_dna": TXTLResourceandMaturationDNAScaffold(),  # M5
    "txtl_model7_bg_fixed":             TXTLModel7_BgFixed(),                # M7 (+ per-reagent learned background)
    "txtl_model8_bg_fixed":             TXTLModel8_BgFixed(),                # M8 (+ per-resource learned background)
    "txtl_model9_event_dark":           TXTLModel9_EventDark(),              # M9 (event-gated dark->fluor)
    # ── Synthetic benchmark scaffolds ──────────────────────────────────────
    "mof_synthesis_12":     MOFSynthesis12Scaffold(),
    "mof_synthesis_8":      MOFSynthesis8Scaffold(),
    "mof_synthesis_6":      MOFSynthesis6Scaffold(),
    "mof_synthesis_4":      MOFSynthesis4Scaffold(),
    "single_enzyme_6":      SingleEnzymeScaffold(),
    "single_enzyme_4":      SingleEnzymeReduced4Scaffold(),
    "single_enzyme_lumped": SingleEnzymeLumpedScaffold(),
    "glycolysis_oracle22":  GlycolysisOracle22Scaffold(),
    "glycolysis_reduced12": GlycolysisReduced12Scaffold(),
    "glycolysis_reduced8":  GlycolysisReduced8Scaffold(),
    "glycolysis_reduced4":  GlycolysisReduced4Scaffold(),
}
