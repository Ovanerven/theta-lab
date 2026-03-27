import numpy as np


def MOF_Synthesis(t: float, y: np.ndarray, k: np.ndarray, dim: bool = False):
    """
    12-state MOF synthesis model.

    Models the competing amorphous vs. crystalline MOF formation pathways,
    with modulator-controlled inhibition of crystalline growth.

    States (12):
      0  Met        : reactive metal in solution
      1  LigH       : protonated ligand (acid form)
      2  Lig_minus  : deprotonated ligand (binding-competent)
      3  H_plus     : proton concentration
      4  Base       : base (control input — drives deprotonation)
      5  Mod        : modulator (control input — caps SBUs, inhibits crystalline growth)
      6  SBU        : secondary building unit (free)
      7  SBU_capped : modulator-capped SBU (inactive)
      8  Nuc_A      : amorphous nuclei
      9  Am         : amorphous product
      10 Nuc_C      : crystalline nuclei
      11 MOF_C      : target crystalline MOF

    Parameters (16):
      0  k_deprot  : deprotonation rate (LigH + Base -> Lig- + ...)
      1  k_prot    : reprotonation rate
      2  k_oli     : oligomerization rate (Met + Lig- -> SBU)
      3  k_cap     : capping rate (SBU + Mod -> SBU_capped)
      4  k_uncap   : uncapping rate
      5  K_I       : modulator inhibition constant for crystalline growth
      6  knuc_A    : amorphous nucleation prefactor
      7  kgro_A    : amorphous growth rate
      8  kagg_A    : amorphous aggregation rate
      9  n_A       : amorphous nucleation exponent on SBU
      10 knuc_C    : crystalline nucleation prefactor
      11 kgro_C    : crystalline growth rate
      12 kagg_C    : crystalline aggregation rate
      13 n_C       : crystalline nucleation exponent on SBU
      14 a         : metal exponent in oligomerization
      15 b         : ligand exponent in oligomerization

    Supervisor default values (from MOF_synthesis.py):
      k_deprot=5.0, k_prot=1.0, k_oli=3.0, k_cap=2.0, k_uncap=0.5,
      K_I=0.1, knuc_A=10.0, kgro_A=1.0, kagg_A=1.0, n_A=3.0,
      knuc_C=0.5, kgro_C=4.0, kagg_C=1.0, n_C=1.5, a=1.0, b=1.0
    """
    if dim:
        states = 12
        parameters = 16
        names = [
            "Met", "LigH", "Lig_minus", "H_plus", "Base", "Mod",
            "SBU", "SBU_capped", "Nuc_A", "Am", "Nuc_C", "MOF_C",
        ]
        return states, parameters, names

    y = np.maximum(0.0, y)
    Met, LigH, Lig_minus, H_plus, Base, Mod, SBU, SBU_capped, Nuc_A, Am, Nuc_C, MOF_C = y
    k_deprot, k_prot, k_oli, k_cap, k_uncap, K_I, knuc_A, kgro_A, kagg_A, n_A, knuc_C, kgro_C, kagg_C, n_C, a, b = k

    r_deprot = k_deprot * LigH * Base
    r_prot   = k_prot * Lig_minus * H_plus
    r_oli    = k_oli * (Met ** a) * (Lig_minus ** b)
    r_cap    = k_cap * SBU * Mod
    r_uncap  = k_uncap * SBU_capped
    r_nuc_A  = knuc_A * (SBU ** n_A)
    r_nuc_C  = knuc_C * (SBU ** n_C)
    r_gro_A  = kgro_A * SBU * Am
    r_agg_A  = kagg_A * (Nuc_A ** 2)
    inhibition_factor = K_I / (K_I + Mod + 1e-6)
    r_gro_C  = kgro_C * SBU * MOF_C * inhibition_factor
    r_agg_C  = kagg_C * (Nuc_C ** 2)

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

    return np.array([
        dMet, dLigH, dLig_minus, dH_plus, dBase, dMod,
        dSBU, dSBU_capped, dNuc_A, dAm, dNuc_C, dMOF_C,
    ], dtype=float)
