"""
Analytical integrator for TXTLResourceandMaturationDNAScaffold.

The ODE system is cascade-linear: R and O decay independently as exponentials,
DNA is constant between jumps, and m/mm/p/pm are linear ODEs driven by those
known signals. This gives a closed-form solution via convolution of exponentials.

States: [R, O, m, mm, p, pm, DNA]
Params: [lam, lam_O, VTXmax, kdm, VTLmax, kmt, kmatm]
"""
import torch

from models.ode_rnn import OdeRNN


def _safe(x: torch.Tensor) -> torch.Tensor:
    """Replace near-zero with 1 so masked branches in torch.where stay finite."""
    return torch.where(x.abs() < 1e-7, torch.ones_like(x), x)


def _conv(A: torch.Tensor,
          r_in: torch.Tensor, r_out: torch.Tensor,
          e_in: torch.Tensor, e_out: torch.Tensor,
          dt: torch.Tensor) -> torch.Tensor:
    """
    A * integral_0^dt  exp(-r_out*(dt-s)) * exp(-r_in*s) ds
      = A * (e_in - e_out) / (r_out - r_in)   [generic]
      = A * dt * e_out                          [r_in == r_out]
    """
    diff = r_out - r_in
    is_deg = diff.abs() < 1e-7
    generic = A * (e_in - e_out) / _safe(diff)
    degen   = A * dt * e_out
    return torch.where(is_deg, degen, generic)


def _coeff(A: torch.Tensor, diff: torch.Tensor) -> torch.Tensor:
    """A / diff with near-zero diff mapped to 0 (degenerate-case coefficient)."""
    return torch.where(diff.abs() < 1e-7, torch.zeros_like(A), A / _safe(diff))


def _integ(C: torch.Tensor, r_x: torch.Tensor,
           lam_O: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """C * (1 - exp(-(lam_O + r_x)*dt)) / (lam_O + r_x)  — integral of C*exp(-(lam_O+r_x)*t)."""
    r_tot = (lam_O + r_x).clamp_min(1e-7)
    return C * (1.0 - torch.exp(-r_tot * dt)) / r_tot


def _integ_bleach(C: torch.Tensor, r_x: torch.Tensor,
                  lam_O: torch.Tensor, kbleach: torch.Tensor,
                  e_b: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    C * integral_0^dt exp(-kbleach*(dt-s)) * exp(-(lam_O + r_x)*s) ds

    Generic:    C * (exp(-(lam_O+r_x)*dt) - exp(-kbleach*dt)) / (kbleach - (lam_O+r_x))
    Degenerate: C * dt * exp(-kbleach*dt)             [when kbleach == lam_O+r_x]

    Reduces to _integ when kbleach == 0 (then exp(-kbleach*dt)=1 and the diff
    cancels into (1 - exp(-r_tot*dt))/r_tot).
    """
    rtot = lam_O + r_x
    diff = kbleach - rtot
    e_tot = torch.exp(-rtot * dt)
    is_deg = diff.abs() < 1e-7
    generic = C * (e_tot - e_b) / _safe(diff)
    degen = C * dt * e_b
    return torch.where(is_deg, degen, generic)


def txtl_step(y0: torch.Tensor, theta: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    Exact solution for TXTLResourceandMaturationDNAScaffold over one interval dt.

    y0    : (B, 7) — [R, O, m, mm, p, pm, DNA]
    theta : (B, 7) — [lam, lam_O, VTXmax, kdm, VTLmax, kmt, kmatm]
    dt    : (B,)
    Returns y_new (B, 7), clamped >= 0.
    """
    R0, O0, m0, mm0, p0, pm0, D0 = y0.clamp_min(0.0).unbind(-1)
    lam, lam_O, VTXmax, kdm, VTLmax, kmt, kmatm = theta.unbind(-1)

    beta = kdm + kmatm          # total m-decay rate

    # Precomputed exponentials for each decay rate
    e_lam  = torch.exp(-lam   * dt)
    e_lamO = torch.exp(-lam_O * dt)
    e_beta = torch.exp(-beta  * dt)
    e_kdm  = torch.exp(-kdm   * dt)
    e_kmt  = torch.exp(-kmt   * dt)
    # Combined rates needed for forcing of p
    r_2lam  = lam + lam
    r_lbeta = lam + beta
    r_lkdm  = lam + kdm
    e_2lam  = e_lam * e_lam     # exp(-(2*lam)*dt)
    e_lbeta = e_lam * e_beta    # exp(-(lam+beta)*dt)
    e_lkdm  = e_lam * e_kdm    # exp(-(lam+kdm)*dt)

    # -------------------------------------------------------------------------
    # R, O, DNA: trivial
    # -------------------------------------------------------------------------
    R_new   = R0 * e_lam
    O_new   = O0 * e_lamO
    DNA_new = D0

    # -------------------------------------------------------------------------
    # m(t) = m0*exp(-beta*t) + VTXmax*D0*R0 * conv(lam→beta)
    # -------------------------------------------------------------------------
    VDR   = VTXmax * D0 * R0
    m_new = m0 * e_beta + _conv(VDR, lam, beta, e_lam, e_beta, dt)

    # Decompose m(t) = m_A*exp(-lam*t) + m_B*exp(-beta*t)
    # so downstream forcing can be tracked per-mode.
    m_A = _coeff(VDR, beta - lam)          # coefficient on exp(-lam*t)
    m_B = m0 - m_A                         # coefficient on exp(-beta*t)
    # (In the degen case lam≈beta, _coeff → 0 and m_B → m0; _conv handles m_new correctly.)

    # -------------------------------------------------------------------------
    # mm(t) = mm0*exp(-kdm*t) + kmatm*m_A*conv(lam→kdm) + kmatm*m_B*conv(beta→kdm)
    # -------------------------------------------------------------------------
    mm_new = (mm0 * e_kdm
              + _conv(kmatm * m_A, lam,  kdm, e_lam,  e_kdm, dt)
              + _conv(kmatm * m_B, beta, kdm, e_beta, e_kdm, dt))

    # Decompose mm(t) = mm_A*exp(-lam*t) + mm_B*exp(-beta*t) + mm_C*exp(-kdm*t)
    mm_A = _coeff(kmatm * m_A, kdm - lam)
    mm_B = _coeff(kmatm * m_B, kdm - beta)
    mm_C = mm0 - mm_A - mm_B

    # -------------------------------------------------------------------------
    # p(t): dp/dt = VTLmax*R(t)*(m+mm) - kmt*p
    # Forcing = VTLmax*R0*exp(-lam*t) * [(m_A+mm_A)*exp(-lam*t)
    #                                    + (m_B+mm_B)*exp(-beta*t)
    #                                    + mm_C*exp(-kdm*t)]
    # Three exponential forcing terms with combined rates 2lam, lam+beta, lam+kdm.
    # -------------------------------------------------------------------------
    VR0    = VTLmax * R0
    sA_mA  = m_A + mm_A           # amplitude of exp(-lam*t) mode in (m+mm)
    sA_mB  = m_B + mm_B           # amplitude of exp(-beta*t) mode
    sA_mmC = mm_C                  # amplitude of exp(-kdm*t) mode

    p_new = (p0 * e_kmt
             + _conv(VR0 * sA_mA,  r_2lam,  kmt, e_2lam,  e_kmt, dt)
             + _conv(VR0 * sA_mB,  r_lbeta, kmt, e_lbeta, e_kmt, dt)
             + _conv(VR0 * sA_mmC, r_lkdm,  kmt, e_lkdm,  e_kmt, dt))

    # Decompose p(t) for pm integral:
    # p(t) = p_A*exp(-2lam*t) + p_B*exp(-(lam+beta)*t)
    #       + p_C*exp(-(lam+kdm)*t) + p_D*exp(-kmt*t)
    p_A = _coeff(VR0 * sA_mA,  kmt - r_2lam)
    p_B = _coeff(VR0 * sA_mB,  kmt - r_lbeta)
    p_C = _coeff(VR0 * sA_mmC, kmt - r_lkdm)
    p_D = p0 - p_A - p_B - p_C

    # -------------------------------------------------------------------------
    # pm(t) = pm0 + kmt*O0 * integral_0^dt exp(-lam_O*s) * p(s) ds
    # Each exponential mode p_X*exp(-r_X*t) contributes:
    #   p_X * (1 - exp(-(lam_O+r_X)*dt)) / (lam_O + r_X)
    # All combined rates (lam_O + r_X) are strictly positive so no degeneracy.
    # -------------------------------------------------------------------------
    pm_new = (pm0
              + kmt * O0 * (
                  _integ(p_A, r_2lam,  lam_O, dt)
                + _integ(p_B, r_lbeta, lam_O, dt)
                + _integ(p_C, r_lkdm,  lam_O, dt)
                + _integ(p_D, kmt,     lam_O, dt)
              ))

    return torch.stack(
        [R_new, O_new, m_new, mm_new, p_new, pm_new, DNA_new], dim=-1
    ).clamp_min(0.0)


class TXTLAnalyticalOdeRNN(OdeRNN):
    """
    OdeRNN with the TXTL cascade solved exactly instead of via RK4.

    Inherits the full GRU → log_gamma(theta) → integrate pipeline from OdeRNN;
    only the integration step is replaced. n_substeps is ignored.

    Use with scaffold: TXTLResourceandMaturationDNAScaffold
    Config key:        ode_rnn_txtl
    """

    def _rk4_substeps(
        self,
        y: torch.Tensor,
        dt: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        # dt arrives as (B,) from OdeRNN.forward (before the unsqueeze in the base method)
        return txtl_step(y, theta, dt)


def txtl_step_bleach(y0: torch.Tensor, theta: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    Exact solution for TXTLResourceandMaturationDNABleachScaffold over one interval dt.
    Same as txtl_step but with dpm/dt = O*kmt*p - kbleach*pm.

    y0    : (B, 7) — [R, O, m, mm, p, pm, DNA]
    theta : (B, 8) — [lam, lam_O, VTXmax, kdm, VTLmax, kmt, kmatm, kbleach]
    dt    : (B,)
    """
    R0, O0, m0, mm0, p0, pm0, D0 = y0.clamp_min(0.0).unbind(-1)
    lam, lam_O, VTXmax, kdm, VTLmax, kmt, kmatm, kbleach = theta.unbind(-1)

    beta = kdm + kmatm

    e_lam   = torch.exp(-lam     * dt)
    e_lamO  = torch.exp(-lam_O   * dt)
    e_beta  = torch.exp(-beta    * dt)
    e_kdm   = torch.exp(-kdm     * dt)
    e_kmt   = torch.exp(-kmt     * dt)
    e_kbl   = torch.exp(-kbleach * dt)

    r_2lam  = lam + lam
    r_lbeta = lam + beta
    r_lkdm  = lam + kdm
    e_2lam  = e_lam * e_lam
    e_lbeta = e_lam * e_beta
    e_lkdm  = e_lam * e_kdm

    # R, O, DNA: trivial
    R_new   = R0 * e_lam
    O_new   = O0 * e_lamO
    DNA_new = D0

    # m(t)
    VDR   = VTXmax * D0 * R0
    m_new = m0 * e_beta + _conv(VDR, lam, beta, e_lam, e_beta, dt)
    m_A = _coeff(VDR, beta - lam)
    m_B = m0 - m_A

    # mm(t)
    mm_new = (mm0 * e_kdm
              + _conv(kmatm * m_A, lam,  kdm, e_lam,  e_kdm, dt)
              + _conv(kmatm * m_B, beta, kdm, e_beta, e_kdm, dt))
    mm_A = _coeff(kmatm * m_A, kdm - lam)
    mm_B = _coeff(kmatm * m_B, kdm - beta)
    mm_C = mm0 - mm_A - mm_B

    # p(t)
    VR0    = VTLmax * R0
    sA_mA  = m_A + mm_A
    sA_mB  = m_B + mm_B
    sA_mmC = mm_C
    p_new = (p0 * e_kmt
             + _conv(VR0 * sA_mA,  r_2lam,  kmt, e_2lam,  e_kmt, dt)
             + _conv(VR0 * sA_mB,  r_lbeta, kmt, e_lbeta, e_kmt, dt)
             + _conv(VR0 * sA_mmC, r_lkdm,  kmt, e_lkdm,  e_kmt, dt))
    p_A = _coeff(VR0 * sA_mA,  kmt - r_2lam)
    p_B = _coeff(VR0 * sA_mB,  kmt - r_lbeta)
    p_C = _coeff(VR0 * sA_mmC, kmt - r_lkdm)
    p_D = p0 - p_A - p_B - p_C

    # pm(t) with bleaching:
    #   pm(dt) = pm0*exp(-kbleach*dt)
    #          + kmt*O0 * Σ_X p_X * conv_bleach(lam_O + r_X, kbleach)
    pm_new = (pm0 * e_kbl
              + kmt * O0 * (
                  _integ_bleach(p_A, r_2lam,  lam_O, kbleach, e_kbl, dt)
                + _integ_bleach(p_B, r_lbeta, lam_O, kbleach, e_kbl, dt)
                + _integ_bleach(p_C, r_lkdm,  lam_O, kbleach, e_kbl, dt)
                + _integ_bleach(p_D, kmt,     lam_O, kbleach, e_kbl, dt)
              ))

    return torch.stack(
        [R_new, O_new, m_new, mm_new, p_new, pm_new, DNA_new], dim=-1
    ).clamp_min(0.0)


class TXTLAnalyticalBleachOdeRNN(OdeRNN):
    """
    OdeRNN with TXTL+bleaching cascade solved exactly.

    Use with scaffold: TXTLResourceandMaturationDNABleachScaffold
    Config key:        ode_rnn_txtl_bleach
    """

    def _rk4_substeps(
        self,
        y: torch.Tensor,
        dt: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        return txtl_step_bleach(y, theta, dt)
