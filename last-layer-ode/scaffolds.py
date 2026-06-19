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
}
