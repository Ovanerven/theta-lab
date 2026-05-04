#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 12:00:02 2025

@author: bob-van-sluijs
"""
from __future__ import annotations
from utils import *
from pathlib import Path
import numpy as np
import torch
from torch import nn
import math
from typing import Mapping, Hashable, Sequence, Dict, List, Tuple
import torch.nn.functional as F
import torchcde
import torch._dynamo as dynamo
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


def gamma(raw: torch.Tensor, lo: float, hi: float, tau: float = 2.3, cap: int = 50) -> torch.Tensor:
    # map R → (0,1) with gentler slope
    s = torch.sigmoid(raw / tau)
    log_lo = math.log(lo); log_hi = math.log(hi)
    return torch.exp(log_lo + (log_hi - log_lo) * s)

def build_x0_ext_on(x0: torch.Tensor) -> torch.Tensor:
    """
    Append two gate states initialised to 1 (ON).

    Returns
    -------
    x0_ext : (B, 5)  [m0, p0, p*0, hTX0, hTL0]
    """
    ones = torch.ones_like(x0[:, :1])        # (B,1)  filled with 1
    x0_ext = torch.cat((x0, ones, ones), dim=1)
    return x0_ext


@torch.jit.script
def integrate_vectors_analytical(
    x0: Tensor,             # (B,5)  m0 p0 p*0  hTX0 hTL0
    dna: Tensor,             # (B,K,1)   ΔDNA
    dt: Tensor,             # (B,K)     Δt
    VTX_raw: Tensor,             # (B,K,1)   gate logits
    VTX_max: Tensor,             # (B,K,1)
    kdm: Tensor,             # (B,K,1)
    VTL_raw: Tensor,             # (B,K,1)   gate logits
    VTL_max: Tensor,             # (B,K,1)
    kmt: Tensor,             # (B,K,1),
    gate_rate: float = 10.0        # relaxation speed κ  (1/min)
) -> Tuple[Tensor, Tensor]:
    """
    Five-state IVTT with continuous ON/OFF gates.

        ḣ_TX = κ (σ(raw_TX) − h_TX)
        VTX   = h_TX · VTX_max

        ḣ_TL = κ (σ(raw_TL) − h_TL)
        VTL   = hturn ((p > 0.5).float() - p).detach() + p    # STE mask
_TL · VTL_max
    """
    eps: float = 1e-9
    B, K, _ = dna.shape

    # buffers ----------------------------------------------------------
    m_out = torch.empty((B, K, 1), device=x0.device)
    p_out = torch.empty_like(m_out)
    pm_out = torch.empty_like(m_out)
    hT_out = torch.empty_like(m_out)
    hL_out = torch.empty_like(m_out)

    # init -------------------------------------------------------------
    m_prev = x0[:, 0:1]
    p_prev = x0[:, 1:2]
    pm_prev = x0[:, 2:3]
    hTX = x0[:, 3:4].clamp(0., 1.)
    hTL = x0[:, 4:5].clamp(0., 1.)
    dna_cum = dna.new_zeros((B, 1))

    for k in range(K):
        dt_k = dt[:, k:k+1]                     # (B,1)
        lam = torch.exp(-gate_rate * dt_k)     # e^{-κΔt}

        # -------- update gates analytically ---------------------------
        hTX_inf = torch.sigmoid(VTX_raw[:, k])     # desired level
        hTL_inf = torch.sigmoid(VTL_raw[:, k])

        hTX = hTX_inf + (hTX - hTX_inf) * lam
        hTL = hTL_inf + (hTL - hTL_inf) * lam

        # -------- effective catalytic rates --------------------------
        VTX_eff = hTX * VTX_max[:, k]
        VTL_eff = hTL * VTL_max[:, k]

        # -------- DNA pool -------------------------------------------
        dna_cum = dna_cum + dna[:, k]

        # -------- mRNA -----------------------------------------------
        kdm_k = kdm[:, k]
        m_inf = (VTX_eff * dna_cum) / (kdm_k + eps)
        exp_m = torch.exp(-kdm_k * dt_k)
        m_curr = m_inf + (m_prev - m_inf) * exp_m
        m_curr = m_curr.clamp_min(0.)

        # -------- ∫ m e^{-kmt(t-τ)} dτ for immature protein p -----------
        kmt_k = kmt[:, k]
        eta = torch.exp(-kmt_k * dt_k)
        diff = kdm_k - kmt_k
        same = diff.abs() < eps

        # This integral (int_m_conv) correctly solves for p_curr
        int_m_conv_eq = (m_inf * (1 - eta) / (kmt_k + eps) +
                         (m_prev - m_inf) * dt_k * eta)
        int_m_conv_gen = (m_inf * (1 - eta) / (kmt_k + eps) +
                          (m_prev - m_inf) *
                          (eta - exp_m) / (diff + eps))
        int_m_conv = torch.where(same, int_m_conv_eq, int_m_conv_gen)

        # -------- Immature protein p ----------------------------------
        p_curr = p_prev * eta + VTL_eff * int_m_conv

        # -------- NEW: Integral of total mRNA for mature protein p* -----
        # This term calculates ∫m(τ)dτ from t_{k-1} to t_k, which represents
        # the total amount of mRNA available for translation during the step.
        int_m_total = m_inf * dt_k + \
            (m_prev - m_inf) / (kdm_k + eps) * (1 - exp_m)

        # -------- CHANGED: Mature protein p* update -------------------
        # The correct update rule derived from the conservation law d/dt(p+p*) = VTL*m.
        # It accounts for both maturation of old protein and synthesis of new protein.
        pm_curr = pm_prev + (p_prev - p_curr) + VTL_eff * int_m_total

        # positivity ---------------------------------------------------
        p_curr = p_curr.clamp_min(0.)
        pm_curr = pm_curr.clamp_min(0.)

        # store --------------------------------------------------------
        m_out[:, k], p_out[:, k], pm_out[:, k] = m_curr, p_curr, pm_curr
        hT_out[:, k], hL_out[:, k] = hTX, hTL

        # roll ---------------------------------------------------------
        m_prev, p_prev, pm_prev = m_curr, p_curr, pm_curr

    out = torch.cat((m_out, p_out, pm_out), dim=2)
    params = torch.cat((VTX_max, hT_out, kdm, VTL_max, hL_out, kmt), dim=2)
    return out, params

@torch.jit.script
def integrate_vectors_analytical_R(
        x0: torch.Tensor,  # (B,4)  m0 p0 p*0 R0  (R0 = 1)
        dna: torch.Tensor,  # (B,K,1)
        dt: torch.Tensor,  # (B,K)
        VTX_max: torch.Tensor,  # (B,K,1)
        kdm: torch.Tensor,  # (B,K,1)
        VTL_max: torch.Tensor,  # (B,K,1)
        kmt: torch.Tensor,  # (B,K,1)
        lam: torch.Tensor   # (B,1)   positive decay rate λ
) -> Tuple[torch.Tensor, torch.Tensor]:
    eps: float = 1e-9
    B, K, _ = dna.shape

    # ---------- output buffers ----------
    m_out = torch.empty((B, K, 1), device=x0.device)
    p_out = torch.empty_like(m_out)
    pm_out = torch.empty_like(m_out)
    R_out = torch.empty_like(m_out)
    
    dna_cum_total = torch.cumsum(dna, dim=1)[:, -1, :]  # (B,1) — total DNA at last timepoint
    # ------------ DNA tensor ------------
    # (B,1) — total DNA at last timepoint
    # ---------- initial state -----------
    m_prev = x0[:, 0:1]                     # (B,1)
    p_prev = x0[:, 1:2]
    pm_prev = x0[:, 2:3]
    R_prev = x0[:, 3:4].clamp_min(eps)      # starts at 1
    dna_cum = dna.new_zeros((B, 1))

    for k in range(K):
        dt_k = dt[:, k:k+1]                  # (B,1)

        # ---- resource decay  R_k = R_{k-1} e^{-λ Δt} ----
        rho = torch.exp(-lam[:, k] * dt_k)       # (B,1)
        R_curr = R_prev * rho

        # ---- effective catalytic rates ----
        VTX_eff = R_curr * VTX_max[:, k]     # (B,1)
        VTL_eff = R_curr * VTL_max[:, k]

        # ---- DNA pool ----------------------
        dna_cum = dna_cum_total    # cumulative μM DNA

        # ---- mRNA --------------------------
        kdm_k = kdm[:, k]
        m_inf = (VTX_eff * dna_cum) / (kdm_k + eps)
        exp_m = torch.exp(-kdm_k * dt_k)
        m_curr = m_inf + (m_prev - m_inf) * exp_m
        m_curr = m_curr.clamp_min(0.)

        # ---- ∫ m e^{-k_mat(t-τ)} dτ --------
        kmt_k = kmt[:, k]
        eta = torch.exp(-kmt_k * dt_k)
        diff = kdm_k - kmt_k
        same = diff.abs() < eps

        int_m_eq = (m_inf * (1 - eta) / (kmt_k + eps) +
                    (m_prev - m_inf) * dt_k * eta)
        int_m_gen = (m_inf * (1 - eta) / (kmt_k + eps) +
                     (m_prev - m_inf) * (eta - exp_m) / (diff + eps))
        int_m_conv = torch.where(same, int_m_eq, int_m_gen)

        # ---- immature protein p ------------
        p_curr = p_prev * eta + VTL_eff * int_m_conv

        # ---- ∫ m dτ for mature protein -----
        int_m_total = (m_inf * dt_k +
                       (m_prev - m_inf) / (kdm_k + eps) * (1 - exp_m))

        pm_curr = pm_prev + (p_prev - p_curr) + VTL_eff * int_m_total

        # ---- positivity & store ------------
        p_curr = p_curr.clamp_min(0.)
        pm_curr = pm_curr.clamp_min(0.)

        m_out[:, k], p_out[:, k], pm_out[:, k] = m_curr, p_curr, pm_curr
        R_out[:, k] = R_curr

        # ---- roll state --------------------
        m_prev, p_prev, pm_prev, R_prev = m_curr, p_curr, pm_curr, R_curr

    out = torch.cat((m_out, p_out, pm_out), dim=2)           # (B,K,3)
    params = torch.cat((VTX_max, kdm, VTL_max, kmt, R_out, lam), dim=2)
    return out, params


@torch.jit.script
def integrate_vectors_analytical_R_mRNA_maturation(
        x0: torch.Tensor,      # (B,5)  [m0, mm0, p0, p*0, R0]
        dna: torch.Tensor,     # (B,K,1)
        dt: torch.Tensor,      # (B,K)
        VTX_max: torch.Tensor, # (B,K,1)
        kdm: torch.Tensor,     # (B,K,1)
        VTL_max: torch.Tensor, # (B,K,1)
        kmt: torch.Tensor,     # (B,K,1)   protein maturation rate
        kmatm: torch.Tensor,   # (B,K,1)   mRNA maturation rate (m -> mm)
        lam: torch.Tensor      # (B,K,1)   resource decay λ (expand beforehand if per-seq)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Analytic IVTT with single resource decay and mRNA maturation:

        dR/dt   = -λ R
        dm/dt   = S - (kdm + kmatm) m                 ,  S = R VTX_max [DNA]_cum
        dmm/dt  = kmatm m - kdm mm
        dp/dt   = (R VTL_max) (m + mm) - kmt p
        dpm/dt  = kmt p

    Assumes parameters are frozen on each [t_{k-1}, t_k) and uses the end-of-step
    R_k = R_{k-1} exp(-λ Δt) as the effective constant within the step.

    Returns:
      out    : (B,K,4)  channels [m, mm, p, p*]
      params : (B,K,7)  [VTX_max, kdm, VTL_max, kmt, kmatm, R, lam]
    """
    eps: float = 1e-9
    B, K, _ = dna.shape

    # ---------- output buffers ----------
    m_out  = torch.empty((B, K, 1), device=x0.device, dtype=x0.dtype)
    mm_out = torch.empty_like(m_out)
    p_out  = torch.empty_like(m_out)
    pm_out = torch.empty_like(m_out)
    R_out  = torch.empty_like(m_out)

    # total DNA per sequence (constant within each step, like your simple solver)
    dna_cum_total = torch.cumsum(dna, dim=1)[:, -1, :]  # (B,1)

    # ---------- initial state -----------
    m_prev   = x0[:, 0:1]              # (B,1)
    p_prev   = x0[:, 1:2]              # (B,1)
    pm_prev  = x0[:, 2:3]              # (B,1)
    R_prev   = x0[:, 3:4].clamp_min(eps)  # (B,1)
    mm_prev  = x0[:, 4:5]              # (B,1)

    # (Optional safety) allow lam given as (B,1) by expanding once
    if lam.size(1) == 1:
        lam = lam.expand(B, K, 1)

    for k in range(K):
        dt_k   = dt[:, k:k+1]          # (B,1)

        # ---- per-step params (B,1) ----
        lam_k  = lam[:,     k]         # (B,1)
        kdm_k  = kdm[:,     k]
        km_k   = kmatm[:,   k]
        kmt_k  = kmt[:,     k]
        VTX_k  = VTX_max[:, k]
        VTL_k  = VTL_max[:, k]

        # ---- resource decay  R_k = R_{k-1} e^{-λ Δt} ----
        rho    = torch.exp(-lam_k * dt_k)
        R_curr = R_prev * rho

        # ---- effective rates & source term ----
        VTX_eff = R_curr * VTX_k
        VTL_eff = R_curr * VTL_k
        S       = VTX_eff * dna_cum_total          # (B,1) constant in step

        # ---- m (immature mRNA): dm/dt = S - (kdm+km) m ----
        alpha  = (kdm_k + km_k).clamp_min(eps)
        m_inf  = S / alpha
        exp_a  = torch.exp(-alpha * dt_k)
        m_curr = m_inf + (m_prev - m_inf) * exp_a
        m_curr = m_curr.clamp_min(0.)

        # ---- mm (mature mRNA): dmm/dt = km m - kdm mm (closed form) ----
        exp_d  = torch.exp(-kdm_k * dt_k)          # e^{-kdm Δt}
        exp_mr = torch.exp(-km_k  * dt_k)          # e^{-km   Δt}
        term1  = mm_prev * exp_d
        term2  = m_inf * (km_k / (kdm_k + eps)) * (1. - exp_d)
        term3  = (m_prev - m_inf) * exp_d * (1. - exp_mr)
        mm_curr = term1 + term2 + term3
        mm_curr = mm_curr.clamp_min(0.)

        # ---- total M = m + mm (for protein) obeys dM/dt = S - kdm M ----
        M_prev = m_prev + mm_prev
        M_inf  = S / (kdm_k + eps)
        exp_M  = exp_d                           # e^{-kdm Δt}

        # ---- protein p: dp/dt = VTL_eff * M - kmt p ----
        eta    = torch.exp(-kmt_k * dt_k)        # e^{-kmt Δt}
        diff   = kdm_k - kmt_k
        same   = diff.abs() < 1e-6

        int_M_eq  = (M_inf * (1. - eta) / (kmt_k + eps)
                     + (M_prev - M_inf) * dt_k * eta)
        int_M_gen = (M_inf * (1. - eta) / (kmt_k + eps)
                     + (M_prev - M_inf) * (eta - exp_M) / (diff + eps))
        int_M_conv = torch.where(same, int_M_eq, int_M_gen)

        p_curr = p_prev * eta + VTL_eff * int_M_conv
        p_curr = p_curr.clamp_min(0.)

        # ---- mature protein p*: conservation d/dt(p+p*) = VTL_eff * M ----
        int_M_total = (M_inf * dt_k
                       + (M_prev - M_inf) / (kdm_k + eps) * (1. - exp_M))
        pm_curr = pm_prev + (p_prev - p_curr) + VTL_eff * int_M_total
        pm_curr = pm_curr.clamp_min(0.)

        # ---- store (same style as your simple solver) ----
        m_out[:,  k]  = m_curr
        mm_out[:, k]  = mm_curr
        p_out[:,  k]  = p_curr
        pm_out[:, k]  = pm_curr
        R_out[:,  k]  = R_curr

        # ---- roll ----
        m_prev, mm_prev, p_prev, pm_prev, R_prev = m_curr, mm_curr, p_curr, pm_curr, R_curr

    out = torch.cat((mm_out, p_out, pm_out), dim=2)                      # (B,K,4)
    params = torch.cat((VTX_max, kdm, VTL_max, kmt, kmatm, R_out, lam), dim=2)  # (B,K,7)
    return out, params
                       
@torch.jit.script
def ivtt_step_mRNA_maturation(
    m_prev:  torch.Tensor,   # (B,1) immature m
    mm_prev: torch.Tensor,   # (B,1) mature mRNA (bright)
    p_prev:  torch.Tensor,   # (B,1) immature protein
    pm_prev: torch.Tensor,   # (B,1) mature protein
    R_prev:  torch.Tensor,   # (B,1)
    dt_k:    torch.Tensor,   # (B,1)
    dna_cum_total: torch.Tensor,  # (B,1) constant within step
    lam_k:   torch.Tensor,   # (B,1)
    VTXmax:  torch.Tensor,   # (B,1)
    kdm:     torch.Tensor,   # (B,1)
    VTLmax:  torch.Tensor,   # (B,1)
    kmt:     torch.Tensor,   # (B,1) protein maturation rate
    kmatm:   torch.Tensor,   # (B,1) mRNA maturation rate (m -> mm)
    eps: float = 1e-9
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One analytic IVTT step with mRNA maturation.
    Returns: (m_curr, mm_curr, p_curr, pm_curr, R_curr), each (B,1).
    """
    # --- resource decay (use end-of-step as effective within step) ---
    rho    = torch.exp(-lam_k * dt_k)     # (B,1)
    R_curr = R_prev * rho

    # --- effective catalytic rates and source S ----------------------
    VTX_eff = R_curr * VTXmax
    VTL_eff = R_curr * VTLmax
    S       = VTX_eff * dna_cum_total     # (B,1), constant within step

    # --- m (immature) ------------------------------------------------
    alpha  = (kdm + kmatm).clamp_min(eps) # (B,1)
    m_inf  = S / alpha
    exp_a  = torch.exp(-alpha * dt_k)
    m_curr = torch.clamp_min(m_inf + (m_prev - m_inf) * exp_a, 0.0)

    # --- mm (matured mRNA) ------------------------------------------
    exp_d  = torch.exp(-kdm   * dt_k)     # e^{-kdm Δt}
    exp_mr = torch.exp(-kmatm * dt_k)     # e^{-kmatm Δt}
    term1  = mm_prev * exp_d
    term2  = m_inf * (kmatm / (kdm + eps)) * (1.0 - exp_d)
    term3  = (m_prev - m_inf) * exp_d * (1.0 - exp_mr)
    mm_curr = torch.clamp_min(term1 + term2 + term3, 0.0)

    # --- total mRNA M for protein formulas --------------------------
    M_prev = m_prev + mm_prev
    M_inf  = S / (kdm + eps)
    exp_M  = exp_d

    # --- protein p ---------------------------------------------------
    eta    = torch.exp(-kmt * dt_k)       # e^{-kmt Δt}
    delta  = (kdm - kmt)
    same   = torch.abs(delta) < 1e-6

    int_M_eq  = (M_inf * (1.0 - eta) / (kmt + eps)
                 + (M_prev - M_inf) * dt_k * eta)
    int_M_gen = (M_inf * (1.0 - eta) / (kmt + eps)
                 + (M_prev - M_inf) * (eta - exp_M) / (delta + eps))
    int_M_conv = torch.where(same, int_M_eq, int_M_gen)

    p_curr  = torch.clamp_min(p_prev * eta + VTL_eff * int_M_conv, 0.0)

    # --- mature protein p* (conservation) ---------------------------
    int_M_total = (M_inf * dt_k
                  + (M_prev - M_inf) / (kdm + eps) * (1.0 - exp_M))
    pm_curr = torch.clamp_min(pm_prev + (p_prev - p_curr) + VTL_eff * int_M_total, 0.0)

    return m_curr, mm_curr, p_curr, pm_curr, R_curr



@torch.jit.script
def ivtt_step_R_O_mRNA_maturation(
    m_prev:  torch.Tensor,   # (B,1)
    mm_prev: torch.Tensor,   # (B,1)
    p_prev:  torch.Tensor,   # (B,1)
    pm_prev: torch.Tensor,   # (B,1)
    R_prev:  torch.Tensor,   # (B,1)
    O_prev:  torch.Tensor,   # (B,1)
    dt_k:    torch.Tensor,   # (B,1)
    dna_cum_total: torch.Tensor,  # (B,1)
    lam_k:   torch.Tensor,   # (B,1)
    lam_O_k: torch.Tensor,   # (B,1)
    VTXmax:  torch.Tensor,   # (B,1)
    kdm:     torch.Tensor,   # (B,1)
    VTLmax:  torch.Tensor,   # (B,1)
    kmt:     torch.Tensor,   # (B,1)
    kmatm:   torch.Tensor,   # (B,1)
    eps: float = 1e-9
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One analytic IVTT step with R decay, O decay, and mRNA maturation.

    Equations:
        dR/dt   = -lam R
        dO/dt   = -lam_O O
        dm/dt   = R VTXmax DNA - (kdm + kmatm) m
        dmm/dt  = kmatm m - kdm mm
        dp/dt   = R VTLmax (m + mm) - kmt p
        dpm/dt  = O kmt p

    Returns:
        m_curr, mm_curr, p_curr, pm_curr, R_curr, O_curr
    """

    # --- resource / O decay, using end-of-step values as effective constants ---
    rho_R  = torch.exp(-lam_k * dt_k)
    rho_O  = torch.exp(-lam_O_k * dt_k)

    R_curr = R_prev * rho_R
    O_curr = O_prev * rho_O

    # --- effective catalytic rates and source S ----------------------
    VTX_eff = R_curr * VTXmax
    VTL_eff = R_curr * VTLmax
    O_eff   = O_curr

    S = VTX_eff * dna_cum_total

    # --- m immature mRNA ---------------------------------------------
    alpha  = (kdm + kmatm).clamp_min(eps)
    m_inf  = S / alpha
    exp_a  = torch.exp(-alpha * dt_k)

    m_curr = torch.clamp_min(
        m_inf + (m_prev - m_inf) * exp_a,
        0.0
    )

    # --- mm mature mRNA ----------------------------------------------
    exp_d  = torch.exp(-kdm * dt_k)
    exp_mr = torch.exp(-kmatm * dt_k)

    term1 = mm_prev * exp_d
    term2 = m_inf * (kmatm / (kdm + eps)) * (1.0 - exp_d)
    term3 = (m_prev - m_inf) * exp_d * (1.0 - exp_mr)

    mm_curr = torch.clamp_min(term1 + term2 + term3, 0.0)

    # --- total mRNA M = m + mm ---------------------------------------
    M_prev = m_prev + mm_prev
    M_inf  = S / (kdm + eps)
    exp_M  = exp_d

    # --- immature protein p ------------------------------------------
    eta   = torch.exp(-kmt * dt_k)
    delta = kdm - kmt
    same  = torch.abs(delta) < 1e-6

    int_M_eq = (
        M_inf * (1.0 - eta) / (kmt + eps)
        + (M_prev - M_inf) * dt_k * eta
    )

    int_M_gen = (
        M_inf * (1.0 - eta) / (kmt + eps)
        + (M_prev - M_inf) * (eta - exp_M) / (delta + eps)
    )

    int_M_conv = torch.where(same, int_M_eq, int_M_gen)

    p_curr = torch.clamp_min(
        p_prev * eta + VTL_eff * int_M_conv,
        0.0
    )

    # --- mature protein p*: dpm/dt = O_eff * kmt * p -----------------
    # Since dp/dt = VTL_eff*M - kmt*p:
    # kmt ∫p dt = VTL_eff∫Mdt - (p_curr - p_prev)
    int_M_total = (
        M_inf * dt_k
        + (M_prev - M_inf) / (kdm + eps) * (1.0 - exp_M)
    )

    pm_curr = torch.clamp_min(
        pm_prev + O_eff * (VTL_eff * int_M_total - (p_curr - p_prev)),
        0.0
    )

    return m_curr, mm_curr, p_curr, pm_curr, R_curr, O_curr

@torch.jit.script
def ivtt_step(
    m_prev: torch.Tensor,    # (B,1)
    p_prev: torch.Tensor,    # (B,1)
    pm_prev: torch.Tensor,   # (B,1)
    R_prev: torch.Tensor,    # (B,1)
    dt_k: torch.Tensor,      # (B,1)
    dna_cum_total: torch.Tensor,  # (B,1)  constant over k
    lam_k: torch.Tensor,     # (B,1)
    VTXmax: torch.Tensor,    # (B,1)
    kdm: torch.Tensor,       # (B,1)
    VTLmax: torch.Tensor,    # (B,1)
    kmt: torch.Tensor,       # (B,1)
    eps: float = 1e-9
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One analytic time step of the IVTT model with resource decay R.
    Returns: (m_curr, p_curr, pm_curr, R_curr), each (B,1).
    """
    # resource decay
    rho    = torch.exp(-lam_k * dt_k)        # (B,1)
    R_curr = R_prev * rho

    # effective catalytic rates
    VTX_eff = R_curr * VTXmax
    VTL_eff = R_curr * VTLmax

    # mRNA
    m_inf  = (VTX_eff * dna_cum_total) / (kdm + eps)
    exp_m  = torch.exp(-kdm * dt_k)
    m_curr = m_inf + (m_prev - m_inf) * exp_m
    m_curr = torch.clamp_min(m_curr, 0.0)

    # immature protein p
    eta    = torch.exp(-kmt * dt_k)
    diff   = kdm - kmt
    same   = torch.abs(diff) < eps
    int_m_eq  = (m_inf * (1.0 - eta) / (kmt + eps)
                 + (m_prev - m_inf) * dt_k * eta)
    int_m_gen = (m_inf * (1.0 - eta) / (kmt + eps)
                 + (m_prev - m_inf) * (eta - exp_m) / (diff + eps))
    int_m_conv = torch.where(same, int_m_eq, int_m_gen)
    p_curr = p_prev * eta + VTL_eff * int_m_conv
    p_curr = torch.clamp_min(p_curr, 0.0)

    # mature protein p* (conservation)
    int_m_total = m_inf * dt_k + (m_prev - m_inf) / (kdm + eps) * (1.0 - exp_m)
    pm_curr = pm_prev + (p_prev - p_curr) + VTL_eff * int_m_total
    pm_curr = torch.clamp_min(pm_curr, 0.0)

    return m_curr, p_curr, pm_curr, R_curr


@torch.jit.script
def ivtt_step_mRNA_two_stage_protein(
    m_prev:  torch.Tensor,   # (B,1) immature mRNA
    mm_prev: torch.Tensor,   # (B,1) observed (mature) mRNA
    p_prev:  torch.Tensor,   # (B,1) immature protein
    pi_prev: torch.Tensor,   # (B,1) intermediate protein (new state)
    pm_prev: torch.Tensor,   # (B,1) observed (mature) protein
    R_prev:  torch.Tensor,   # (B,1) resource
    dt_k:    torch.Tensor,   # (B,1)
    dna_cum_total: torch.Tensor,  # (B,1) constant within step
    lam_k:   torch.Tensor,   # (B,1)
    VTXmax:  torch.Tensor,   # (B,1)
    kdm:     torch.Tensor,   # (B,1) mRNA decay
    VTLmax:  torch.Tensor,   # (B,1)
    kmt1:    torch.Tensor,   # (B,1) p -> p_i       (was kmt)
    kmt2:    torch.Tensor,   # (B,1) p_i -> p_m     (new time-dep param)
    kmatm:   torch.Tensor,   # (B,1) m -> mm
    eps: float = 1e-9
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One analytic IVTT step with two-stage protein maturation.
    Returns: (m_curr, mm_curr, p_curr, pi_curr, pm_curr, R_curr), each (B,1).
    """

    # ---------- resource decay ----------
    rho    = torch.exp(-lam_k * dt_k)
    R_curr = R_prev * rho

    # ---------- effective rates ----------
    VTX_eff = R_curr * VTXmax
    VTL_eff = R_curr * VTLmax
    S       = VTX_eff * dna_cum_total   # constant within step

    # ---------- m (immature)  dm/dt = S - (kdm + kmatm) m ----------
    alpha  = (kdm + kmatm).clamp_min(eps)
    m_inf  = S / alpha
    exp_a  = torch.exp(-alpha * dt_k)
    m_curr = torch.clamp_min(m_inf + (m_prev - m_inf) * exp_a, 0.0)

    # ---------- mm (observed mRNA) ----------
    exp_d  = torch.exp(-kdm   * dt_k)     # e^{-kdm Δt}
    exp_mr = torch.exp(-kmatm * dt_k)     # e^{-kmatm Δt}
    term1  = mm_prev * exp_d
    term2  = m_inf * (kmatm / (kdm + eps)) * (1.0 - exp_d)
    term3  = (m_prev - m_inf) * exp_d * (1.0 - exp_mr)
    mm_curr = torch.clamp_min(term1 + term2 + term3, 0.0)

    # ---------- total mRNA M for protein terms ----------
    M_prev = m_prev + mm_prev
    M_inf  = S / (kdm + eps)
    exp_M  = exp_d  # e^{-kdm Δt}

    # ---------- p (immature protein): dp/dt = VTL_eff M - kmt1 p ----------
    eta1   = torch.exp(-kmt1 * dt_k)     # e^{-kmt1 Δt}
    delta1 = (kdm - kmt1)
    same1  = torch.abs(delta1) < 1e-6

    # ∫_0^Δt M(τ) e^{-kmt1 (Δt-τ)} dτ  (your existing single-stage integral)
    int_M_eq_1  = (M_inf * (1.0 - eta1) / (kmt1 + eps)
                   + (M_prev - M_inf) * dt_k * eta1)
    int_M_gen_1 = (M_inf * (1.0 - eta1) / (kmt1 + eps)
                   + (M_prev - M_inf) * (eta1 - exp_M) / (delta1 + eps))
    I1 = torch.where(same1, int_M_eq_1, int_M_gen_1)

    p_curr  = torch.clamp_min(p_prev * eta1 + VTL_eff * I1, 0.0)

    # ---------- p_i (intermediate): dpi/dt = kmt1 p - kmt2 p_i ----------
    eta2   = torch.exp(-kmt2 * dt_k)     # e^{-kmt2 Δt}
    diff12 = (kmt1 - kmt2)
    same12 = torch.abs(diff12) < 1e-6

    # Initial p contribution to p_i (closed form)
    # J1 = kmt1 * p_prev * e^{-kmt2 Δt} * (1 - e^{-(kmt1-kmt2) Δt})/(kmt1 - kmt2)
    J1_gen = (kmt1 / (diff12 + eps)) * (eta2 - eta1) * p_prev
    # Equal-rate limit: J1 = k * dt * e^{-k dt} * p_prev
    J1_eq  = (kmt1 * dt_k * eta1) * p_prev
    J1     = torch.where(same12, J1_eq, J1_gen)

    # Input M contribution to p_i via two-stage kernel:
    # J2 = kmt1 * VTL_eff * [ I2 - I1 ] / (kmt1 - kmt2),
    # where I2 is the single-stage integral with kmt2
    delta2 = (kdm - kmt2)
    same2  = torch.abs(delta2) < 1e-6
    int_M_eq_2  = (M_inf * (1.0 - eta2) / (kmt2 + eps)
                   + (M_prev - M_inf) * dt_k * eta2)
    int_M_gen_2 = (M_inf * (1.0 - eta2) / (kmt2 + eps)
                   + (M_prev - M_inf) * (eta2 - exp_M) / (delta2 + eps))
    I2 = torch.where(same2, int_M_eq_2, int_M_gen_2)

    J2_gen = (kmt1 / (diff12 + eps)) * VTL_eff * (I2 - I1)

    # Equal-rate limit for J2: k * dI/dk at k=kmt1 (finite-diff fallback)
    # Small positive delta for numerical derivative:
    delta_k = torch.full_like(kmt1, 1e-6)
    k_plus  = (kmt1 + delta_k).clamp_min(eps)
    eta_plus = torch.exp(-k_plus * dt_k)
    delta_plus = (kdm - k_plus)
    same_plus  = torch.abs(delta_plus) < 1e-6
    int_M_eq_p = (M_inf * (1.0 - eta_plus) / (k_plus + eps)
                  + (M_prev - M_inf) * dt_k * eta_plus)
    int_M_gen_p= (M_inf * (1.0 - eta_plus) / (k_plus + eps)
                  + (M_prev - M_inf) * (eta_plus - exp_M) / (delta_plus + eps))
    I_plus = torch.where(same_plus, int_M_eq_p, int_M_gen_p)
    dI_dk  = (I_plus - I1) / (delta_k + eps)
    J2_eq  = kmt1 * VTL_eff * dI_dk

    J2 = torch.where(same12, J2_eq, J2_gen)

    pi_curr = torch.clamp_min(pi_prev * eta2 + J1 + J2, 0.0)

    # ---------- p_m (observed mature protein) via conservation ----------
    # Total protein increases by VTL_eff * ∫ M(τ) dτ
    int_M_total = (M_inf * dt_k
                   + (M_prev - M_inf) / (kdm + eps) * (1.0 - exp_M))
    pm_curr = torch.clamp_min(pm_prev + (p_prev + pi_prev - (p_curr + pi_curr)) + VTL_eff * int_M_total, 0.0)

    return m_curr, mm_curr, p_curr, pi_curr, pm_curr, R_curr



@torch.jit.script
def integrate_vectors_analytical_R_O_mRNA_maturation(
        x0: torch.Tensor,       # (B,6) [m0, p0, p*0, R0, mm0, O0]
        dna: torch.Tensor,      # (B,K,1)
        dt: torch.Tensor,       # (B,K)
        VTX_max: torch.Tensor,  # (B,K,1)
        kdm: torch.Tensor,      # (B,K,1)
        VTL_max: torch.Tensor,  # (B,K,1)
        kmt: torch.Tensor,      # (B,K,1)
        kmatm: torch.Tensor,    # (B,K,1)
        lam: torch.Tensor,      # (B,K,1) or (B,1)
        lam_O: torch.Tensor     # (B,K,1) or (B,1)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Analytic IVTT with resource decay R, output/resource factor O, and mRNA maturation:

        dR/dt   = -lam R
        dO/dt   = -lam_O O
        dm/dt   = R VTX_max DNA - (kdm + kmatm) m
        dmm/dt  = kmatm m - kdm mm
        dp/dt   = R VTL_max (m + mm) - kmt p
        dpm/dt  = O kmt p

    Parameters are frozen per step. R and O use end-of-step values as effective
    constants within each step.
    """
    eps: float = 1e-9
    B, K, _ = dna.shape

    m_out  = torch.empty((B, K, 1), device=x0.device, dtype=x0.dtype)
    mm_out = torch.empty_like(m_out)
    p_out  = torch.empty_like(m_out)
    pm_out = torch.empty_like(m_out)
    R_out  = torch.empty_like(m_out)
    O_out  = torch.empty_like(m_out)

    dna_cum_total = torch.cumsum(dna, dim=1)[:, -1, :]  # (B,1)

    m_prev  = x0[:, 0:1]
    p_prev  = x0[:, 1:2]
    pm_prev = x0[:, 2:3]
    R_prev  = x0[:, 3:4].clamp_min(eps)
    mm_prev = x0[:, 4:5]
    O_prev  = x0[:, 5:6].clamp_min(eps)

    if lam.size(1) == 1:
        lam = lam.expand(B, K, 1)

    if lam_O.size(1) == 1:
        lam_O = lam_O.expand(B, K, 1)

    for k in range(K):
        dt_k = dt[:, k:k+1]

        lam_k   = lam[:, k]
        lam_O_k = lam_O[:, k]

        kdm_k = kdm[:, k]
        km_k  = kmatm[:, k]
        kmt_k = kmt[:, k]
        VTX_k = VTX_max[:, k]
        VTL_k = VTL_max[:, k]

        # ---- R and O decay ----
        R_curr = R_prev * torch.exp(-lam_k * dt_k)
        O_curr = O_prev * torch.exp(-lam_O_k * dt_k)

        # ---- effective stepwise constants ----
        VTX_eff = R_curr * VTX_k
        VTL_eff = R_curr * VTL_k
        O_eff   = O_curr

        S = VTX_eff * dna_cum_total

        # ---- m ----
        alpha = (kdm_k + km_k).clamp_min(eps)
        m_inf = S / alpha
        exp_a = torch.exp(-alpha * dt_k)

        m_curr = m_inf + (m_prev - m_inf) * exp_a
        m_curr = m_curr.clamp_min(0.0)

        # ---- mm ----
        exp_d  = torch.exp(-kdm_k * dt_k)
        exp_mr = torch.exp(-km_k * dt_k)

        term1 = mm_prev * exp_d
        term2 = m_inf * (km_k / (kdm_k + eps)) * (1.0 - exp_d)
        term3 = (m_prev - m_inf) * exp_d * (1.0 - exp_mr)

        mm_curr = term1 + term2 + term3
        mm_curr = mm_curr.clamp_min(0.0)

        # ---- total mRNA M = m + mm ----
        M_prev = m_prev + mm_prev
        M_inf  = S / (kdm_k + eps)
        exp_M  = exp_d

        # ---- p ----
        eta  = torch.exp(-kmt_k * dt_k)
        diff = kdm_k - kmt_k
        same = diff.abs() < 1e-6

        int_M_eq = (
            M_inf * (1.0 - eta) / (kmt_k + eps)
            + (M_prev - M_inf) * dt_k * eta
        )

        int_M_gen = (
            M_inf * (1.0 - eta) / (kmt_k + eps)
            + (M_prev - M_inf) * (eta - exp_M) / (diff + eps)
        )

        int_M_conv = torch.where(same, int_M_eq, int_M_gen)

        p_curr = p_prev * eta + VTL_eff * int_M_conv
        p_curr = p_curr.clamp_min(0.0)

        # ---- p*: dpm/dt = O_eff * kmt * p ----
        # Using: kmt ∫p dt = VTL_eff ∫M dt - (p_curr - p_prev)
        int_M_total = (
            M_inf * dt_k
            + (M_prev - M_inf) / (kdm_k + eps) * (1.0 - exp_M)
        )

        pm_increment = O_eff * (
            VTL_eff * int_M_total - (p_curr - p_prev)
        )

        pm_curr = pm_prev + pm_increment
        pm_curr = pm_curr.clamp_min(0.0)

        # ---- store ----
        m_out[:, k]  = m_curr
        mm_out[:, k] = mm_curr
        p_out[:, k]  = p_curr
        pm_out[:, k] = pm_curr
        R_out[:, k]  = R_curr
        O_out[:, k]  = O_curr

        # ---- roll ----
        m_prev  = m_curr
        mm_prev = mm_curr
        p_prev  = p_curr
        pm_prev = pm_curr
        R_prev  = R_curr
        O_prev  = O_curr

    out = torch.cat((m_out, mm_out, p_out, pm_out), dim=2)

    params = torch.cat(
        (VTX_max, kdm, VTL_max, kmt, kmatm, R_out, O_out, lam, lam_O),
        dim=2
    )

    return out, params
    

# --------------- Closed-loop stacked-GRU with scripted step ------------------
class GRU_model_latent_decay_simple_fb_matO(nn.Module):
    """
    Closed-loop, stacked GRUCells with state feedback:
      input at step k: [u_k, dna_k, dt_k, log1p(mm_{k-1}), log1p(p*_{k-1})]
      -> stacked GRUCells
      -> θ_k = [λ, VTX_max, kdm, VTL_max, kmt, kmatm]  (bounded)
      -> ivtt_step_mRNA_maturation() -> states for next step

    Returns:
      out    : (B,K,3) with channels [mm, p, p*]
      params : (B,K,7) with [VTX_max, kdm, VTL_max, kmt, kmatm, R, lam]
    """
    def __init__(self,
                 in_u: int,
                 hidden: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2 ):
        super().__init__()
        assert num_layers >= 1
        self.hidden = hidden
        self.num_layers = num_layers

        in_dim = in_u+2
        # stacked GRUCells
        cells = [nn.GRUCell(in_dim, hidden)]
        cells += [nn.GRUCell(hidden, hidden) for _ in range(num_layers - 1)]
        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout(dropout)
        
        self.bottle = nn.Sequential(nn.Linear(hidden,  128), nn.SiLU(),
                                    nn.Linear(128,  64), nn.SiLU(),   
                                    )
        self.head = nn.Linear(64, 7)  # [lam, VTX_mag, kdm, VTL_mag, kmt, kmatm]
                

        for c in self.cells:
            nn.init.orthogonal_(c.weight_hh)
            nn.init.xavier_uniform_(c.weight_ih)
            nn.init.zeros_(c.bias_hh)
            nn.init.zeros_(c.bias_ih)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self,
                x0:      torch.Tensor,   # (B,2): [m_obs0, p*0]  (obs mRNA is matured)
                u_seq:   torch.Tensor,   # (B,K,U)
                dna_raw: torch.Tensor,   # (B,K,1)
                dt_seq:  torch.Tensor,   # (B,K)
                y_seq:   torch.Tensor,
                teacher_forcing: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:

        B, K, U = u_seq.shape
        dev = u_seq.device
        dtype = u_seq.dtype

        # outputs (we return mm, p, pm)
        mm_out = torch.empty(B, K, 1, device=dev, dtype=dtype)
        p_out  = torch.empty_like(mm_out)
        pm_out = torch.empty_like(mm_out)

        # diagnostics
        VTX_seq = torch.empty_like(mm_out)
        kdm_seq = torch.empty_like(mm_out)
        VTL_seq = torch.empty_like(mm_out)
        kmt_seq = torch.empty_like(mm_out)
        kmatm_seq = torch.empty_like(mm_out)
        R_seq   = torch.empty_like(mm_out)
        lam_seq = torch.empty_like(mm_out)
        lamO_seq = torch.empty_like(mm_out)


        mm_prev = x0[:, 0:1]  + 0.01                # measured mRNA is mature
        m_prev  = torch.zeros_like(mm_prev) + 0.01 # start immature m at 0
        pm_prev = x0[:, 2:3]  + 0.01 
        p_prev  = torch.zeros_like(mm_prev)  + 0.01 
        R_prev  = torch.ones_like(mm_prev)
        O_prev  = torch.ones_like(mm_prev)
        
        # constant DNA per sequence (matches integrator)
        dna_cum_total = torch.cumsum(dna_raw, dim=1)[:, -1, :]  # (B,1)

        # hidden states
        hs = [torch.zeros(B, self.hidden, device=dev, dtype=dtype) for _ in range(self.num_layers)]
        y_mm = y_seq[:, :, 0:1]  # (B,K,1)
        y_pm = y_seq[:, :, 2:3]  # (B,K,1)
        
        for k in range(K):
            dt_k  = dt_seq[:, k:k+1]      # (B,1)
            u_k   = u_seq[:, k]           # (B,U)
            mm_gt_prev = mm_prev
            pm_gt_prev = pm_prev
            
            if k == 0:
                # for step 0, use the provided initial obs x0
                mm_gt_prev = mm_prev   # mature mRNA
                pm_gt_prev = pm_prev  # mature protein
  

            if k%200 == 0 and teacher_forcing == True:
                # use ground-truth from previous step if available
                mm_gt_prev = y_mm[:, k-1, :] 
                pm_gt_prev = y_pm[:, k-1, :] 
                
            det_mm = mm_gt_prev.detach()
            det_pm = pm_gt_prev.detach()
            feat_k = torch.cat([torch.sqrt(u_k).clamp_min(0),
                                torch.sqrt(det_mm).clamp_min(1),
                                torch.sqrt(det_pm).clamp_min(1),
                                # torch.sqrt(R_prev),
                                # torch.sqrt(O_prev),
                                ]
                               , dim=-1)  # (B, in_dim) 
            
            x = feat_k
            for li, cell in enumerate(self.cells):
                hs[li] = cell(x, hs[li])
                x = self.dropout(hs[li]) if self.training else hs[li]
            h = x  # (B, hidden)

            # head → raw params
            z = self.bottle(h)
            raw = self.head(z)
            lam_raw,lamO_raw, VTX_mag, kdm_raw, VTL_mag, kmt_raw, kmatm_raw = raw.split(1, dim=-1)

                 
            lam_k = gamma(lam_raw, 1e-6, 0.0005)
            lam_O = gamma(lamO_raw, 1e-6, 0.0005)
            VTXmax = gamma(VTX_mag, 5e-5, 0.1)
            VTLmax = gamma(VTL_mag, 5e-5, 0.06)
            kdm_k = gamma(kdm_raw, 1e-5, 1e-2)
            kmt_k = gamma(kmt_raw, 1e-5, 0.00035)
            kmatm_k = gamma(kmatm_raw, 5e-5, 0.0035)
       
            
            # ---- one scripted step ----

            m_curr, mm_curr, p_curr, pm_curr, R_curr, O_curr = ivtt_step_R_O_mRNA_maturation(
                m_prev, mm_prev, p_prev, pm_prev, R_prev,O_prev,
                dt_k, dna_cum_total,
                lam_k,lam_O, VTXmax, kdm_k, VTLmax, kmt_k, kmatm_k)
            
            # store
            mm_out[:, k:k+1, :]   = mm_curr.unsqueeze(1)
            p_out[:,  k:k+1,  :]  = p_curr.unsqueeze(1)
            pm_out[:, k:k+1,  :]  = pm_curr.unsqueeze(1)

            VTX_seq[:,  k:k+1, :] = VTXmax.unsqueeze(1)
            kdm_seq[:,  k:k+1, :] = kdm_k.unsqueeze(1)
            VTL_seq[:,  k:k+1, :] = VTLmax.unsqueeze(1)
            kmt_seq[:,  k:k+1, :] = kmt_k.unsqueeze(1)
            kmatm_seq[:,k:k+1, :] = kmatm_k.unsqueeze(1)
            R_seq[:,    k:k+1, :] = R_curr.unsqueeze(1)
            lam_seq[:,  k:k+1, :] = lam_k.unsqueeze(1)
            lamO_seq[:,  k:k+1, :] = lam_O.unsqueeze(1)

            # roll
            m_prev, mm_prev, p_prev, pm_prev, R_prev , O_prev = m_curr, mm_curr, p_curr, pm_curr, R_curr, O_curr

        out    = torch.cat([mm_out, p_out, pm_out], dim=-1)  # (B,K,3): [mm, p, p*]
        params = torch.cat([VTX_seq, kdm_seq, VTL_seq, kmt_seq, kmatm_seq, R_seq, lam_seq, lamO_seq], dim=-1)  # (B,K,7)

        return (out, params) if self.training else (out, params.detach())