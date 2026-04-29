import torch
import torch.nn as nn
import math

class MechanisticScaffold(nn.Module):
    def __init__(self, P: int, theta_dim: int):
        super().__init__()
        self.P = int(P)
        self.theta_dim = int(theta_dim)
        self.state_names: list[str] = []
        # Per-parameter bounds — set by subclasses. None means use scalar fallback.
        self.theta_lo_vec: "list[float] | None" = None
        self.theta_hi_vec: "list[float] | None" = None

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

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

class TXTLMaturationDNAScaffold(MechanisticScaffold):
    """
    6-state TXTL scaffold with DNA as an explicit, bolus-driven state.

    The mechanism is the supervisor's `TXTL_mRNAMaturation`, with DNA promoted
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
        super().__init__(P=6, theta_dim=6)
        self.state_names = ["R", "m", "mm", "p", "pm", "DNA"]
        # Supervisor's log-uniform bounds for TXTL_mRNAMaturation
        self.theta_lo_vec = [1e-6, 3e-5, 1e-5, 3e-5, 1e-5, 5e-5]
        self.theta_hi_vec = [5e-4, 1.2e-1, 1e-2, 8e-2, 3.5e-4, 3.5e-3]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        R, m, mm, p, pm, DNA = y.unbind(dim=-1)
        lam, VTXmax, kdm, VTLmax, kmt, kmatm = theta.unbind(dim=-1)

        R_p   = torch.clamp_min(R,   0.0)
        m_p   = torch.clamp_min(m,   0.0)
        mm_p  = torch.clamp_min(mm,  0.0)
        p_p   = torch.clamp_min(p,   0.0)
        DNA_p = torch.clamp_min(DNA, 0.0)

        dR   = -lam * R_p
        dm   = R_p * VTXmax * DNA_p - (kdm + kmatm) * m_p
        dmm  = kmatm * m_p - kdm * mm_p
        dp   = R_p * VTLmax * (m_p + mm_p) - kmt * p_p
        dpm  = kmt * p_p
        dDNA = torch.zeros_like(DNA)

        return torch.stack((dR, dm, dmm, dp, dpm, dDNA), dim=-1)

class TXTLResourceandMaturationDNAScaffold(MechanisticScaffold):
    """
    6-state TXTL scaffold with DNA as an explicit, bolus-driven state.

    The mechanism is the supervisor's `TXTL_mRNAMaturation`, with DNA promoted
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
        # Supervisor's log-uniform bounds for TXTL_mRNAMaturation
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


class TXTLResourceandMaturationDNABleachScaffold(MechanisticScaffold):
    """
    Extension of TXTLResourceandMaturationDNAScaffold with a pm bleaching term:
        dpm/dt = O * kmt * p - kbleach * pm

    Motivation: real IVTT pm trajectories on failure runs *decrease* over time
    (mCherry photobleaching during long readouts). The base scaffold's dpm is
    non-negative, so it cannot represent this — best it can do is plateau pm.
    Adding kbleach as an 8th learned parameter lets the encoder pull pm down on
    samples where the truth declines, without affecting samples where it rises.

    States (7): R, O, m, mm, p, pm, DNA  (same as base)

    Parameters θ (8):
      0  lam     : resource decay rate
      1  lam_O   : oxygen decay rate
      2  VTXmax  : transcription rate
      3  kdm     : mRNA degradation rate
      4  VTLmax  : translation rate
      5  kmt     : protein maturation rate (p → pm)
      6  kmatm   : mRNA maturation rate (m → mm)
      7  kbleach : pm decay rate (photobleaching / measurement decay)
    """
    def __init__(self):
        super().__init__(P=7, theta_dim=8)
        self.state_names = ["R", "O", "m", "mm", "p", "pm", "DNA"]
        self.theta_lo_vec = [1e-6, 1e-6, 3e-5, 1e-5, 3e-5, 1e-5, 5e-5, 1e-7]
        self.theta_hi_vec = [5e-4, 5e-4, 1.2e-1, 1e-2, 8e-2, 3.5e-4, 3.5e-3, 1e-4]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        R, O, m, mm, p, pm, DNA = y.unbind(dim=-1)
        lam, lam_O, VTXmax, kdm, VTLmax, kmt, kmatm, kbleach = theta.unbind(dim=-1)

        R_p   = torch.clamp_min(R,   0.0)
        O_p   = torch.clamp_min(O,   0.0)
        m_p   = torch.clamp_min(m,   0.0)
        mm_p  = torch.clamp_min(mm,  0.0)
        p_p   = torch.clamp_min(p,   0.0)
        pm_p  = torch.clamp_min(pm,  0.0)
        DNA_p = torch.clamp_min(DNA, 0.0)

        dR   = -lam * R_p
        dO   = -lam_O * O_p
        dm   = R_p * VTXmax * DNA_p - (kdm + kmatm) * m_p
        dmm  = kmatm * m_p - kdm * mm_p
        dp   = R_p * VTLmax * (m_p + mm_p) - kmt * p_p
        dpm  = O_p * kmt * p_p - kbleach * pm_p
        dDNA = torch.zeros_like(DNA)

        return torch.stack((dR, dO, dm, dmm, dp, dpm, dDNA), dim=-1)


class TXTLSimpleDNAScaffold(MechanisticScaffold):
    """
    3-state minimal TXTL scaffold with DNA as an explicit, bolus-driven state.

    The simplest cascade DNA → mm → pm with first-order kinetics. No resource
    pool, no mRNA maturation, no protein maturation — the network must learn
    time-varying θ(t) to compensate for the missing structure.

    States (3): mm (Broccoli), pm (mCherry / 2), DNA

    Parameters θ (3):
      0  k_tx : transcription rate (DNA → mm)
      1  k_tl : translation rate (mm → pm)
      2  kdm  : mRNA degradation rate

    Observed indices within P: [0, 1]  (mm, pm)
    Use with: datasets/real_ivtt_simple.npz (layout='simple')
    """
    def __init__(self):
        super().__init__(P=3, theta_dim=3)
        self.state_names = ["mm", "pm", "DNA"]
        self.theta_lo_vec = [1e-5, 1e-5, 1e-5]
        self.theta_hi_vec = [1e-1, 1e-1, 1e-2]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        mm, pm, DNA = y.unbind(dim=-1)
        k_tx, k_tl, kdm = theta.unbind(dim=-1)

        mm_p  = torch.clamp_min(mm,  0.0)
        DNA_p = torch.clamp_min(DNA, 0.0)

        dmm  = k_tx * DNA_p - kdm * mm_p
        dpm  = k_tl * mm_p
        dDNA = torch.zeros_like(DNA)

        return torch.stack((dmm, dpm, dDNA), dim=-1)

class MethaneGlobal4Step_NO_Scaffold(MechanisticScaffold):
    """
    A physically grounded 4-step macroscopic scaffold for Methane oxidation.
    Instead of a 49-parameter black box, we only learn 4 kinetic parameters 
    representing the main branches of combustion.
    
    States (7): CH4, O2, CO, CO2, H2O, OH, NO
    Parameters (4):
      0: k_methane_ox : CH4 -> CO + H2O (Partial oxidation)
      1: k_co_ox      : CO -> CO2       (CO burnout)
      2: k_oh_prod    : O2 + H2O -> OH  (Radical pool generation)
      3: k_thermal_no : O2 -> NO        (Thermal NO formation proxy)
    """
    def __init__(self):
        super().__init__(P=7, theta_dim=4) # Dropped from 49 to 4!
        self.state_names = ["CH4", "O2", "CO", "CO2", "H2O", "OH", "NO"]
        
        # Bounding the 4 reaction rates
        self.theta_lo_vec = [1e-5, 1e-5, 1e-5, 1e-6]
        self.theta_hi_vec = [10.0, 10.0, 10.0, 1.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        CH4, O2, CO, CO2, H2O, OH, NO = y.unbind(dim=-1)
        k_methane, k_co, k_oh, k_no = theta.unbind(dim=-1)

        # Clamp to prevent negative concentrations causing runaway physics
        CH4_p = torch.clamp_min(CH4, 0.0)
        O2_p  = torch.clamp_min(O2,  0.0)
        CO_p  = torch.clamp_min(CO,  0.0)
        H2O_p = torch.clamp_min(H2O, 0.0)

        # Ensure rates are positive
        k_methane = torch.clamp_min(k_methane, 0.0)
        k_co      = torch.clamp_min(k_co, 0.0)
        k_oh      = torch.clamp_min(k_oh, 0.0)
        k_no      = torch.clamp_min(k_no, 0.0)

        # Calculate fluxes for the 4 macroscopic steps (using simple mass action / linear rates)
        # 1. CH4 + 1.5 O2 -> CO + 2 H2O
        r1 = k_methane * CH4_p * O2_p 
        
        # 2. CO + 0.5 O2 -> CO2
        r2 = k_co * CO_p * O2_p
        
        # 3. O2 -> 2 OH (Conceptual radical formation)
        r3 = k_oh * O2_p
        
        # 4. N2 + O2 -> 2 NO (N2 is assumed constant in air, so rate just depends on O2)
        r4 = k_no * O2_p

        # Apply stoichiometry to state derivatives
        dCH4 = -r1
        dO2  = -1.5 * r1 - 0.5 * r2 - r3 - r4
        dCO  =  r1 - r2
        dCO2 =  r2
        dH2O =  2.0 * r1
        dOH  =  2.0 * r3
        dNO  =  2.0 * r4

        return torch.stack((dCH4, dO2, dCO, dCO2, dH2O, dOH, dNO), dim=-1)

class MethaneGlobal4Step_CH2O_Scaffold(MechanisticScaffold):
    """
    A physically grounded 4-step macroscopic scaffold for the Smooke methane model.
    Routes carbon explicitly through the CH2O intermediate.
    
    States (7): CH4, O2, CO, CO2, H2O, OH, CH2O
    Parameters (4):
      0: k_methane : CH4 + O2 -> CH2O + H2O      (Methane to Formaldehyde)
      1: k_ch2o    : CH2O + 0.5 O2 -> CO + H2O   (Formaldehyde to CO)
      2: k_co      : CO + 0.5 O2 -> CO2          (CO burnout)
      3: k_oh      : O2 -> 2 OH                  (Radical pool proxy)
    """
    def __init__(self):
        super().__init__(P=7, theta_dim=4)
        self.state_names = ["CH4", "O2", "CO", "CO2", "H2O", "OH", "CH2O"]
        
        self.theta_lo_vec = [1e-5, 1e-5, 1e-5, 1e-6]
        self.theta_hi_vec = [10.0, 10.0, 10.0, 1.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        CH4, O2, CO, CO2, H2O, OH, CH2O = y.unbind(dim=-1)
        k_methane, k_ch2o, k_co, k_oh = theta.unbind(dim=-1)

        # Clamp states to prevent negative concentrations
        CH4_p  = torch.clamp_min(CH4, 0.0)
        O2_p   = torch.clamp_min(O2,  0.0)
        CO_p   = torch.clamp_min(CO,  0.0)
        CH2O_p = torch.clamp_min(CH2O, 0.0)

        # Clamp rates to prevent reverse physics
        k_methane = torch.clamp_min(k_methane, 0.0)
        k_ch2o    = torch.clamp_min(k_ch2o, 0.0)
        k_co      = torch.clamp_min(k_co, 0.0)
        k_oh      = torch.clamp_min(k_oh, 0.0)

        # 1. CH4 -> CH2O
        r1 = k_methane * CH4_p * O2_p 
        
        # 2. CH2O -> CO
        r2 = k_ch2o * CH2O_p * O2_p
        
        # 3. CO -> CO2
        r3 = k_co * CO_p * O2_p
        
        # 4. OH generation proxy
        r4 = k_oh * O2_p

        # Apply mass-balanced stoichiometry to the derivatives
        dCH4  = -r1
        dO2   = -r1 - 0.5 * r2 - 0.5 * r3 - r4
        dCO   =  r2 - r3
        dCO2  =  r3
        dH2O  =  r1 + r2
        dOH   =  2.0 * r4
        dCH2O =  r1 - r2

        return torch.stack((dCH4, dO2, dCO, dCO2, dH2O, dOH, dCH2O), dim=-1)


class MethaneDomainInformedCH2O_OHGate4Step_Scaffold(MechanisticScaffold):
        """
        Domain-informed 4-step CH2O scaffold with OH-gated CH2O oxidation.

        States (7): CH4, O2, CO, CO2, H2O, OH, CH2O
        Parameters (4):
            0: k_methane : CH4 + O2 -> CH2O + H2O
            1: k_ch2o    : CH2O + OH -> CO + H2O + H (OH-gated)
            2: k_co      : CO + 0.5 O2 -> CO2
            3: k_oh      : O2 -> 2 OH
        """
        def __init__(self):
                super().__init__(P=7, theta_dim=4)
                self.state_names = ["CH4", "O2", "CO", "CO2", "H2O", "OH", "CH2O"]

                self.theta_lo_vec = [1e-5, 1e-5, 1e-5, 1e-6]
                self.theta_hi_vec = [10.0, 10.0, 10.0, 1.0]

        def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
                CH4, O2, CO, CO2, H2O, OH, CH2O = y.unbind(dim=-1)
                k_methane, k_ch2o, k_co, k_oh = theta.unbind(dim=-1)

                CH4_p  = torch.clamp_min(CH4, 0.0)
                O2_p   = torch.clamp_min(O2,  0.0)
                CO_p   = torch.clamp_min(CO,  0.0)
                CH2O_p = torch.clamp_min(CH2O, 0.0)
                OH_p   = torch.clamp_min(OH,  0.0)

                k_methane = torch.clamp_min(k_methane, 0.0)
                k_ch2o    = torch.clamp_min(k_ch2o, 0.0)
                k_co      = torch.clamp_min(k_co, 0.0)
                k_oh      = torch.clamp_min(k_oh, 0.0)

                r1 = k_methane * CH4_p * O2_p
                r2 = k_ch2o * CH2O_p * (OH_p / (OH_p + 1e-4))
                r3 = k_co * CO_p * O2_p
                r4 = k_oh * O2_p

                dCH4  = -r1
                dO2   = -r1 - 0.5 * r3 - r4
                dCO   = r2 - r3
                dCO2  = r3
                dH2O  = r1 + r2
                dOH   = 2.0 * r4
                dCH2O = r1 - r2

                return torch.stack((dCH4, dO2, dCO, dCO2, dH2O, dOH, dCH2O), dim=-1)
    
class MethaneDomainInformedOHGate4Step_NO_Scaffold(MechanisticScaffold):
    """
    Domain-informed macroscopic scaffold for Methane oxidation.
    Incorporates reversibility, water-assisted CO oxidation, and fractional exponents.
    
    States (7): CH4, O2, CO, CO2, H2O, OH, NO
    Parameters (8):
      0: k_methane_ox : CH4 forward oxidation
      1: n_o2_methane : Fractional order of O2 in CH4 oxidation
      2: k_co_f       : CO -> CO2 forward rate
      3: k_co_r       : CO2 -> CO reverse rate (Equilibrium bottleneck)
      4: k_wgs        : Water-gas shift proxy (CO + H2O -> CO2)
      5: k_oh_prod    : Radical pool generation
      6: k_thermal_no : NO formation
      7: n_o2_no      : Fractional order of O2 in NO formation (Thermal NO is highly non-linear)
    """
    def __init__(self):
        super().__init__(P=7, theta_dim=8)
        self.state_names = ["CH4", "O2", "CO", "CO2", "H2O", "OH", "NO"]
        
        # Bounds: rates can be wide, exponents bounded between ~0.1 and 2.0
        self.theta_lo_vec = [1e-5, 0.1, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 0.1]
        self.theta_hi_vec = [10.0, 2.0, 10.0, 10.0, 10.0, 10.0, 1.0,  2.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        CH4, O2, CO, CO2, H2O, OH, NO = y.unbind(dim=-1)
        (k_methane, n_o2_methane, k_co_f, k_co_r, 
         k_wgs, k_oh, k_no, n_o2_no) = theta.unbind(dim=-1)

        # Clamp states to prevent negative concentrations and NaN powers
        eps = 1e-8
        CH4_p = torch.clamp_min(CH4, eps)
        O2_p  = torch.clamp_min(O2,  eps)
        CO_p  = torch.clamp_min(CO,  eps)
        CO2_p = torch.clamp_min(CO2, eps)
        H2O_p = torch.clamp_min(H2O, eps)
        OH_p  = torch.clamp_min(OH,  0.0)

        # Clamp parameters to bounds
        k_methane    = torch.clamp_min(k_methane, 0.0)
        n_o2_methane = torch.clamp(n_o2_methane, min=0.1, max=2.0)
        k_co_f       = torch.clamp_min(k_co_f, 0.0)
        k_co_r       = torch.clamp_min(k_co_r, 0.0)
        k_wgs        = torch.clamp_min(k_wgs, 0.0)
        k_oh         = torch.clamp_min(k_oh, 0.0)
        k_no         = torch.clamp_min(k_no, 0.0)
        n_o2_no      = torch.clamp(n_o2_no, min=0.1, max=2.0)

        # 1. CH4 Oxidation (with learned fractional O2 dependence + OH gating)
        oh_gate = OH_p / (OH_p + 1e-3)
        r1 = k_methane * CH4_p * (O2_p ** n_o2_methane) * oh_gate
        
        # 2. Reversible CO Burnout + Water Gas Shift Proxy
        # r2_f: CO + 0.5 O2 -> CO2
        # r2_r: CO2 -> CO + 0.5 O2
        # r_wgs: CO + H2O -> CO2 + (hidden)
        r2_f  = k_co_f * CO_p * (O2_p ** 0.5)
        r2_r  = k_co_r * CO2_p
        r_wgs = k_wgs * CO_p * H2O_p
        
        # 3. OH Generation
        r3 = k_oh * O2_p
        
        # 4. Thermal NO formation (highly sensitive to O2)
        r4 = k_no * (O2_p ** n_o2_no)

        # Apply stoichiometry
        dCH4 = -r1
        dO2  = -1.5 * r1 - 0.5 * r2_f + 0.5 * r2_r - r3 - r4
        dCO  =  r1 - r2_f + r2_r - r_wgs
        dCO2 =  r2_f - r2_r + r_wgs
        dH2O =  2.0 * r1 - r_wgs
        dOH  =  2.0 * r3
        dNO  =  2.0 * r4

        return torch.stack((dCH4, dO2, dCO, dCO2, dH2O, dOH, dNO), dim=-1)
    
class MethaneRevWGS_OHGate4Step_NO_Scaffold(MechanisticScaffold):
    """
    Advanced Domain-informed macroscopic scaffold.
    Fixes the CO2 overshoot problem by replacing O2-driven CO burnout 
    with OH-gated CO burnout, mirroring the true CO + OH <-> CO2 + H reaction.
    Also introduces a reversible Water-Gas Shift proxy.
    
    States (7): CH4, O2, CO, CO2, H2O, OH, NO
    Parameters (8):
      0: k_methane_ox : CH4 forward oxidation
      1: n_o2_methane : Fractional order of O2 in CH4 oxidation
      2: k_co_oh      : CO -> CO2 forward rate (GATED BY OH)
      3: k_co_r       : CO2 -> CO reverse rate 
      4: k_wgs_f      : Water-gas shift forward (CO + H2O -> CO2 + ...)
      5: k_wgs_r      : Water-gas shift reverse (CO2 -> CO + H2O proxy)
      6: k_oh_prod    : Radical pool generation
      7: k_thermal_no : NO formation
    """
    def __init__(self):
        super().__init__(P=7, theta_dim=8)
        self.state_names = ["CH4", "O2", "CO", "CO2", "H2O", "OH", "NO"]
        
        # Bounds optimized for the new kinetic formulation
        self.theta_lo_vec = [1e-5, 0.1, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6]
        self.theta_hi_vec = [10.0, 2.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        CH4, O2, CO, CO2, H2O, OH, NO = y.unbind(dim=-1)
        (k_methane, n_o2_methane, k_co_oh, k_co_r, 
         k_wgs_f, k_wgs_r, k_oh, k_no) = theta.unbind(dim=-1)

        # Clamp states to prevent negative concentrations and NaN powers
        eps = 1e-8
        CH4_p = torch.clamp_min(CH4, eps)
        O2_p  = torch.clamp_min(O2,  eps)
        CO_p  = torch.clamp_min(CO,  eps)
        CO2_p = torch.clamp_min(CO2, eps)
        H2O_p = torch.clamp_min(H2O, eps)
        OH_p  = torch.clamp_min(OH,  eps)

        # Clamp parameters to bounds
        k_methane    = torch.clamp_min(k_methane, 0.0)
        n_o2_methane = torch.clamp(n_o2_methane, min=0.1, max=2.0)
        k_co_oh      = torch.clamp_min(k_co_oh, 0.0)
        k_co_r       = torch.clamp_min(k_co_r, 0.0)
        k_wgs_f      = torch.clamp_min(k_wgs_f, 0.0)
        k_wgs_r      = torch.clamp_min(k_wgs_r, 0.0)
        k_oh         = torch.clamp_min(k_oh, 0.0)
        k_no         = torch.clamp_min(k_no, 0.0)

        # 1. CH4 Oxidation (Requires OH pool to truly kick off - induction proxy)
        # We add a mild OH saturation term to force the ignition delay
        r1 = k_methane * CH4_p * (O2_p ** n_o2_methane) * (OH_p / (OH_p + 1e-3))
        
        # 2. Reversible CO Burnout (GATED BY OH instead of O2)
        # r2_f: CO + OH -> CO2 + H (proxy)
        r2_f = k_co_oh * CO_p * OH_p
        r2_r = k_co_r * CO2_p
        
        # 3. Fully Reversible Water-Gas Shift
        r_wgs_f = k_wgs_f * CO_p * H2O_p
        r_wgs_r = k_wgs_r * CO2_p  # We don't track H2, so we approximate the reverse rate linearly
        r_wgs_net = r_wgs_f - r_wgs_r
        
        # 4. OH Generation (Fuel inhibition proxy: early CH4 suppresses OH accumulation)
        r3 = k_oh * O2_p / (1.0 + 10.0 * CH4_p)
        
        # 5. NO formation
        r4 = k_no * O2_p

        # Apply stoichiometry
        dCH4 = -r1
        dO2  = -1.5 * r1 - r3 - r4
        dCO  =  r1 - r2_f + r2_r - r_wgs_net
        dCO2 =  r2_f - r2_r + r_wgs_net
        dH2O =  2.0 * r1 - r_wgs_net
        dOH  =  2.0 * r3 - r2_f  # OH is consumed during CO burnout
        dNO  =  2.0 * r4

        return torch.stack((dCH4, dO2, dCO, dCO2, dH2O, dOH, dNO), dim=-1)

SCAFFOLDS: dict[str, MechanisticScaffold] = {
    "mof_synthesis_12":  MOFSynthesis12Scaffold(),
    "mof_synthesis_8":   MOFSynthesis8Scaffold(),
    "mof_synthesis_6":   MOFSynthesis6Scaffold(),
    "mof_synthesis_4":   MOFSynthesis4Scaffold(),
    "single_enzyme_6":   SingleEnzymeScaffold(),
    "single_enzyme_4":   SingleEnzymeReduced4Scaffold(),
    "single_enzyme_lumped": SingleEnzymeLumpedScaffold(),
    "txtl_maturation_dna": TXTLMaturationDNAScaffold(),
    "txtl_simple_dna":     TXTLSimpleDNAScaffold(),
    "txtl_resource_and_maturation_dna": TXTLResourceandMaturationDNAScaffold(),
    "methane_global4_no":    MethaneGlobal4Step_NO_Scaffold(),
    "methane_global4_ch2o":  MethaneGlobal4Step_CH2O_Scaffold(),
    "methane_domain4_ch2o_ohgate": MethaneDomainInformedCH2O_OHGate4Step_Scaffold(),
    "methane_domain4_no_ohgate": MethaneDomainInformedOHGate4Step_NO_Scaffold(),
    "methane_revWGS_ohgate_no": MethaneRevWGS_OHGate4Step_NO_Scaffold(),
    "txtl_resource_and_maturation_dna_bleach": TXTLResourceandMaturationDNABleachScaffold(),
}
