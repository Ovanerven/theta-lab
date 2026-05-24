def _pos(x: float | np.ndarray) -> float | np.ndarray:
    """Positive part."""
    return np.maximum(x, 0.0)

def _safe_log(x: float | np.ndarray, eps: float = 1e-12) -> float | np.ndarray:
    return np.log(np.maximum(x, eps))

def _hill(x: float, n: float, K: float, eps: float = 1e-12) -> float:
    """Hill saturation x^n / (K^n + x^n)."""
    x = max(float(x), 0.0)
    K = max(float(K), eps)
    xn = x**n
    return xn / (K**n + xn + eps)

def _cnt_like_nucleation(S: float, kJ: float, Bcnt: float, p: float = 0.0, eps: float = 1e-12) -> float:
    """
    CNT-shaped nucleation rate: kJ * exp( -Bcnt / (ln S)^2 ) * (S-1)^p  for S>1 else 0
    """
    if S <= 1.0 + 1e-12:
        return 0.0
    lnS = float(_safe_log(S, eps=eps))
    return float(kJ * np.exp(-Bcnt / (lnS*lnS + eps)) * (S - 1.0)**p)

def _surface_area_spheres(N: float, R: float, shape_factor: float = 1.0) -> float:
    """Surface area per volume for monodisperse spheres: SA = shape_factor * 4*pi*R^2*N."""
    return float(shape_factor * 4.0 * np.pi * max(R, 0.0)**2 * max(N, 0.0))

def _volume_spheres(N: float, R: float, shape_factor: float = 1.0) -> float:
    """Solid volume fraction proxy (per suspension volume): V = shape_factor * (4/3)*pi*R^3*N."""
    return float(shape_factor * (4.0/3.0) * np.pi * max(R, 0.0)**3 * max(N, 0.0))


def MOF_Mechanistic_Max(t: float, y: np.ndarray, k: np.ndarray, dim: bool = False) -> np.ndarray:
    """
    A "maximal" mechanistic MOF formation model intended for nonlinear dynamical benchmarks.

    State vector (18):
      0  M      : reactive metal species in solution (effective concentration)
      1  LH     : protonated linker (acid form)
      2  Lm     : deprotonated linker (binding-competent)
      3  H      : proton concentration proxy (not full charge balance)
      4  BH     : protonated modulator (acid form, e.g., formic/acetic acid)
      5  Bm     : deprotonated modulator (binding-competent competitor)
      6  C1     : ML complex
      7  C2     : ML2 complex (dominant growth unit precursor)
      8  C3     : ML3 complex (off-pathway / overcoordinated)
      9  CB     : MB competitor complex (metal trapped by modulator)
      10 P      : prenucleation cluster / SBU pool (effective "growth unit")
      11 A      : amorphous / dense-liquid intermediate
      12 N      : number density of crystalline nuclei / crystals
      13 R      : mean crystal size (radius proxy)
      14 D      : defect fraction proxy (0..1; modulator-driven missing-linker/cluster tendency)
      15 T      : temperature (K) (constant unless you 'bolus' it)
      16 Solv   : solvent quality / water activity proxy (constant unless bolused) (affects Peq)
      17 I      : inert impurity/poison proxy (inhibits nucleation/growth; constant unless bolused)

    Parameters (44) packed as k[0:44]:
    """
    if dim:
        states = 18
        parameters = 44
        names = ["M","LH","Lm","H","BH","Bm","C1","C2","C3","CB","P","A","N","R","D","T","Solv","I"]
        return states, parameters, names

    # Unpack states
    M, LH, Lm, H, BH, Bm, C1, C2, C3, CB, P, A, N, R, D, T, Solv, I = [float(v) for v in y]

    # Unpack parameters
    (ka_L, kma_L, ka_B, kma_B,
     k1f, k1r, k2f, k2r, k3f, k3r,
     kBf, kBr,
     nu, kp_f, kp_r,
     m, kcond, alphaA,
     nA, KA, kcryst,
     Peq0, Eeq, betaH, betaB, betaS, Tmin, Tmax,
     kJ, Bcnt, pJ, ksec, qsec, shape, kagg,
     kg, gG, KiB, kdiss, chi,
     kDf, KD, kDa, EaD) = [float(v) for v in k]

    _R_GAS = 1
    # Clamp temperature to avoid numerical blow-up if user boluses T to nonsense
    T_eff = min(max(T, Tmin), Tmax)

    # --- Acid/base (proxy; not electroneutrality enforced) ---
    r_L_deprot = ka_L * LH
    r_L_reprot = kma_L * Lm * H

    r_B_deprot = ka_B * BH
    r_B_reprot = kma_B * Bm * H

    # --- Complexation / speciation (mass-action) ---
    r1f = k1f * M * Lm
    r1r = k1r * C1

    r2f = k2f * C1 * Lm
    r2r = k2r * C2

    r3f = k3f * C2 * Lm
    r3r = k3r * C3

    # --- Competitor binding ---
    rBf = kBf * M * Bm
    rBr = kBr * CB

    # --- Cluster / SBU pool (effective) ---
    # Allow non-integer 'nu' to tune nonlinearity; guard against negative.
    C2_pos = max(C2, 0.0)
    nu_eff = max(nu, 1e-6)
    rPf = kp_f * (C2_pos ** nu_eff)
    rPr = kp_r * max(P, 0.0)

    # --- Condensation to amorphous (autocatalytic) ---
    P_pos = max(P, 0.0)
    A_pos = max(A, 0.0)
    m_eff = max(m, 1e-6)
    r_cond = kcond * (P_pos ** m_eff) * (1.0 + alphaA * A_pos)

    # --- Crystallization from amorphous (saturating) ---
    r_cryst = kcryst * _hill(A_pos, n=max(nA, 1e-6), K=max(KA, 1e-12))

    # --- Supersaturation with environment dependence ---
    # Peq(T, H, Bm, Solv): Arrhenius-like temperature dependence + linear modifiers.
    Peq_T = Peq0 * np.exp(-Eeq / (_R_GAS * T_eff + 1e-12))
    Peq = Peq_T * (1.0 + betaH * max(H, 0.0)) * (1.0 + betaB * max(Bm, 0.0)) * (1.0 + betaS * max(Solv, 0.0))
    Peq = float(max(Peq, 1e-12))
    S = float(P_pos / Peq)

    # --- Geometry (surface area) ---
    SA = _surface_area_spheres(N, R, shape_factor=max(shape, 0.0))

    # --- Nucleation: primary (CNT-like) + secondary (surface-catalyzed) ---
    J_prim = _cnt_like_nucleation(S, kJ=max(kJ, 0.0), Bcnt=max(Bcnt, 0.0), p=max(pJ, 0.0))
    J_sec = max(ksec, 0.0) * SA * (_pos(S - 1.0) ** max(qsec, 0.0))

    # Impurity/poison suppression (simple exponential)
    poison = np.exp(-max(I, 0.0))
    J_prim *= poison
    J_sec *= poison

    # --- Growth / dissolution ---
    inhib = 1.0 / (1.0 + max(KiB, 0.0) * max(Bm, 0.0))
    G_pos = max(kg, 0.0) * (_pos(S - 1.0) ** max(gG, 0.0)) * inhib * poison
    G_neg = max(kdiss, 0.0) * (_pos(1.0 - S))  # dissolution rate magnitude
    dR = G_pos - G_neg

    # Growth-unit consumption/return (only surface-driven part)
    r_grow_consume = max(chi, 0.0) * SA * max(G_pos, 0.0)
    r_grow_return  = max(chi, 0.0) * SA * max(G_neg, 0.0)

    # --- N aggregation / coalescence loss ---
    r_agg = max(kagg, 0.0) * max(N, 0.0)**2

    # --- Defect dynamics (phenomenological; 0..1) ---
    d_sat = _hill(max(Bm, 0.0), n=1.0, K=max(KD, 1e-12))
    r_D_form = max(kDf, 0.0) * d_sat * (1.0 - np.clip(D, 0.0, 1.0))
    r_D_ann  = max(kDa, 0.0) * np.clip(D, 0.0, 1.0) * np.exp(-max(EaD, 0.0)/(_R_GAS*T_eff + 1e-12))

    # ======================
    # ODEs (mass balances)
    # ======================

    # M balance: consumed by linker complexation and competitor binding; regenerated by reverse steps
    dM = (-r1f + r1r) + (-rBf + rBr)

    # Linker acid-base
    dLH = -r_L_deprot + r_L_reprot
    dLm = +r_L_deprot - r_L_reprot

    # Modulator acid-base
    dBH = -r_B_deprot + r_B_reprot
    dBm = +r_B_deprot - r_B_reprot

    # Proton proxy
    dH = (+r_L_deprot - r_L_reprot) + (+r_B_deprot - r_B_reprot)

    # Speciation ladder
    dC1 = (+r1f - r1r) + (-r2f + r2r)
    dC2 = (+r2f - r2r) + (-r3f + r3r) + (-nu_eff*rPf + nu_eff*rPr)
    dC3 = (+r3f - r3r)

    # Competitor complex
    dCB = (+rBf - rBr)

    # Cluster pool P: formed from C2, lost to condensation, consumed by growth, replenished by dissolution.
    dP = (+rPf - rPr) - r_cond - r_grow_consume + r_grow_return

    # Amorphous intermediate: formed by condensation, crystallizes, can redissolve back to P when undersaturated
    # (simple: redissolve proportional to A and undersaturation)
    r_A_rediss = 0.1 * max(kdiss, 0.0) * A_pos * _pos(1.0 - S)
    dA = (+r_cond) - r_cryst - r_A_rediss

    # Crystal population: N increases by nucleation (plus a small fraction from crystallization flux),
    # decreases by aggregation/occlusion.
    dN = (J_prim + J_sec + 0.05*r_cryst) - r_agg

    # Mean size
    dR_dt = dR

    # Defects
    dD_dt = r_D_form - r_D_ann

    # Environment states (constant unless user boluses them)
    dT = 0.0
    dSolv = 0.0
    dI = 0.0

    # Couple solution species consumption to solid formation (optional stoichiometric coupling)
    # Here we implement a minimal coupling: crystallization consumes linker-metal material from P, not directly from M/Lm,
    # because P is our effective growth unit pool. Users can calibrate kp_f/kp_r and chi to enforce depletion.
    # If you WANT explicit metal/linker depletion, couple r_grow_consume to C2 or to M/Lm in your own variant.

    return np.array([dM, dLH, dLm, dH, dBH, dBm, dC1, dC2, dC3, dCB, dP, dA, dN, dR_dt, dD_dt, dT, dSolv, dI], dtype=float)


# =============================================================================
# 2) ZIF-8-LIKE THREE-STEP NUCLEATION (Liu et al. PNAS 2021) 
# =============================================================================
def MOF_ZIF8_ThreeStep(t: float, y: np.ndarray, k: np.ndarray, dim: bool = False) -> np.ndarray:
    """
    ZIF-8-inspired three-step nonclassical nucleation: phase separation -> amorphous -> crystal.

    States (8):
      M  : metal proxy (e.g., Zn reactive)
      L  : linker proxy (e.g., mIm- reactive)
      P  : solute-rich condensed droplets / dense clusters (phase-separated)
      A  : amorphous aggregate
      N  : nuclei / crystals number density
      R  : mean crystal size
      S  : supersaturation proxy state (optional, can be driven/bolused)
      T  : temperature (K)

    Parameters (18):
      0 kML      M+L -> P   (effective formation of dense phase; can be high order)
      1 aML      exponent on M
      2 bML      exponent on L
      3 kLP      P -> M+L   (redissolution of dense phase)
      4 kPA      P -> A     (condensation)
      5 kAP      A -> P     (reverse)
      6 kAN      A -> nuclei flux (crystallization)
      7 nA       Hill exponent for A->N
      8 KA       Hill half-sat for A->N
      9 kJ       primary nucleation prefactor (optional CNT-like on S)
      10 Bcnt    CNT barrier parameter
      11 kg      growth prefactor
      12 gG      growth exponent
      13 kdiss   dissolution prefactor (undersaturated)
      14 chi     consumption factor
      15 shape   surface shape factor
      16 kagg    aggregation loss for N
      17 Peq0    equilibrium P proxy for S = P/Peq0

    This is a compact form you can fit to in-situ data where you can infer
    intermediate populations (P,A) qualitatively.
    """
    if dim:
        states = 8
        parameters = 18
        names = ["M","L","P","A","N","R","S","T"]
        return states, parameters, names

    M, L, P, A, N, R, S_state, T = [float(v) for v in y]
    (kML, aML, bML, kLP, kPA, kAP, kAN, nA, KA,
     kJ, Bcnt, kg, gG, kdiss, chi, shape, kagg, Peq0) = [float(v) for v in k]

    # Effective phase separation/condensation: M+L -> P with general order
    r_ML = max(kML, 0.0) * (max(M,0.0)**max(aML,0.0)) * (max(L,0.0)**max(bML,0.0))
    r_LP = max(kLP, 0.0) * max(P,0.0)

    r_PA = max(kPA, 0.0) * max(P,0.0)
    r_AP = max(kAP, 0.0) * max(A,0.0)

    r_AN = max(kAN, 0.0) * _hill(max(A,0.0), n=max(nA,1e-6), K=max(KA,1e-12))

    # Supersaturation proxy: either use explicit state S_state (driven by user) or compute from P/Peq0
    Peq0 = max(Peq0, 1e-12)
    S = max(P,0.0)/Peq0
    # Blend with state to allow user to "impose" measured supersaturation:
    S_eff = 0.5*S + 0.5*max(S_state, 0.0)

    SA = _surface_area_spheres(N, R, shape_factor=max(shape, 0.0))
    J = _cnt_like_nucleation(S_eff, kJ=max(kJ,0.0), Bcnt=max(Bcnt,0.0), p=0.0)

    G_pos = max(kg,0.0) * (_pos(S_eff-1.0)**max(gG,0.0))
    G_neg = max(kdiss,0.0) * (_pos(1.0-S_eff))
    dR = G_pos - G_neg

    consume = max(chi,0.0) * SA * max(G_pos,0.0)
    returnP = max(chi,0.0) * SA * max(G_neg,0.0)

    dM = -r_ML + r_LP
    dL = -r_ML + r_LP
    dP = +r_ML - r_LP - r_PA + r_AP - consume + returnP
    dA = +r_PA - r_AP - r_AN
    dN = +J + 0.1*r_AN - max(kagg,0.0)*max(N,0.0)**2
    dR_dt = dR
    dS = 0.0  # user-driven
    dT = 0.0
    return np.array([dM,dL,dP,dA,dN,dR_dt,dS,dT], dtype=float)
