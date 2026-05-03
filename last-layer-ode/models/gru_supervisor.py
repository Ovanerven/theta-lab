"""Bob's supervisor architecture, copied verbatim from configs_oliver so that
GRU_O_mat_6000.pt / GRU_O_mat_6005.pt load with strict=True.

Do not refactor: the parameter names, layer order, and split layout matter for
state-dict compatibility.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def gamma(raw: torch.Tensor, lo: float, hi: float, tau: float = 2.3) -> torch.Tensor:
    s = torch.sigmoid(raw / tau)
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    return torch.exp(log_lo + (log_hi - log_lo) * s)


@torch.jit.script
def ivtt_step_R_O_mRNA_maturation(
    m_prev: torch.Tensor,
    mm_prev: torch.Tensor,
    p_prev: torch.Tensor,
    pm_prev: torch.Tensor,
    R_prev: torch.Tensor,
    O_prev: torch.Tensor,
    dt_k: torch.Tensor,
    dna_cum_total: torch.Tensor,
    lam_k: torch.Tensor,
    lam_O_k: torch.Tensor,
    VTXmax: torch.Tensor,
    kdm: torch.Tensor,
    VTLmax: torch.Tensor,
    kmt: torch.Tensor,
    kmatm: torch.Tensor,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rho_R = torch.exp(-lam_k * dt_k)
    rho_O = torch.exp(-lam_O_k * dt_k)

    R_curr = R_prev * rho_R
    O_curr = O_prev * rho_O

    VTX_eff = R_curr * VTXmax
    VTL_eff = R_curr * VTLmax
    O_eff = O_curr

    S = VTX_eff * dna_cum_total

    alpha = (kdm + kmatm).clamp_min(eps)
    m_inf = S / alpha
    exp_a = torch.exp(-alpha * dt_k)

    m_curr = torch.clamp_min(m_inf + (m_prev - m_inf) * exp_a, 0.0)

    exp_d = torch.exp(-kdm * dt_k)
    exp_mr = torch.exp(-kmatm * dt_k)

    term1 = mm_prev * exp_d
    term2 = m_inf * (kmatm / (kdm + eps)) * (1.0 - exp_d)
    term3 = (m_prev - m_inf) * exp_d * (1.0 - exp_mr)

    mm_curr = torch.clamp_min(term1 + term2 + term3, 0.0)

    M_prev = m_prev + mm_prev
    M_inf = S / (kdm + eps)
    exp_M = exp_d

    eta = torch.exp(-kmt * dt_k)
    delta = kdm - kmt
    same = torch.abs(delta) < 1e-6

    int_M_eq = (
        M_inf * (1.0 - eta) / (kmt + eps)
        + (M_prev - M_inf) * dt_k * eta
    )
    int_M_gen = (
        M_inf * (1.0 - eta) / (kmt + eps)
        + (M_prev - M_inf) * (eta - exp_M) / (delta + eps)
    )
    int_M_conv = torch.where(same, int_M_eq, int_M_gen)

    p_curr = torch.clamp_min(p_prev * eta + VTL_eff * int_M_conv, 0.0)

    int_M_total = (
        M_inf * dt_k
        + (M_prev - M_inf) / (kdm + eps) * (1.0 - exp_M)
    )

    pm_curr = torch.clamp_min(
        pm_prev + O_eff * (VTL_eff * int_M_total - (p_curr - p_prev)),
        0.0,
    )

    return m_curr, mm_curr, p_curr, pm_curr, R_curr, O_curr


class GRU_model_latent_decay_simple_fb_matO(nn.Module):
    """Closed-loop stacked-GRUCell with state feedback. State-dict compatible
    with GRU_O_mat_6000.pt / GRU_O_mat_6005.pt (in_u=12, hidden=400, layers=2)."""

    def __init__(self, in_u: int, hidden: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        assert num_layers >= 1
        self.hidden = hidden
        self.num_layers = num_layers

        in_dim = in_u + 2
        cells = [nn.GRUCell(in_dim, hidden)]
        cells += [nn.GRUCell(hidden, hidden) for _ in range(num_layers - 1)]
        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout(dropout)

        self.bottle = nn.Sequential(
            nn.Linear(hidden, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
        )
        self.head = nn.Linear(64, 7)

        for c in self.cells:
            nn.init.orthogonal_(c.weight_hh)
            nn.init.xavier_uniform_(c.weight_ih)
            nn.init.zeros_(c.bias_hh)
            nn.init.zeros_(c.bias_ih)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x0: torch.Tensor,        # (B,3): [mm0, p0, pm0]
        u_seq: torch.Tensor,     # (B,K,U) — pass already-normalized u (e.g. u/u_max)
        dna_raw: torch.Tensor,   # (B,K,1)
        dt_seq: torch.Tensor,    # (B,K)
        y_seq: torch.Tensor,     # (B,K,3) — only used when teacher_forcing
        teacher_forcing: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, K, U = u_seq.shape
        dev = u_seq.device
        dtype = u_seq.dtype

        mm_out = torch.empty(B, K, 1, device=dev, dtype=dtype)
        p_out = torch.empty_like(mm_out)
        pm_out = torch.empty_like(mm_out)

        VTX_seq = torch.empty_like(mm_out)
        kdm_seq = torch.empty_like(mm_out)
        VTL_seq = torch.empty_like(mm_out)
        kmt_seq = torch.empty_like(mm_out)
        kmatm_seq = torch.empty_like(mm_out)
        R_seq = torch.empty_like(mm_out)
        lam_seq = torch.empty_like(mm_out)
        lamO_seq = torch.empty_like(mm_out)

        mm_prev = x0[:, 0:1] + 0.01
        m_prev = torch.zeros_like(mm_prev) + 0.01
        pm_prev = x0[:, 2:3] + 0.01
        p_prev = torch.zeros_like(mm_prev) + 0.01
        R_prev = torch.ones_like(mm_prev)
        O_prev = torch.ones_like(mm_prev)

        dna_cum_total = torch.cumsum(dna_raw, dim=1)[:, -1, :]

        hs = [torch.zeros(B, self.hidden, device=dev, dtype=dtype) for _ in range(self.num_layers)]
        y_mm = y_seq[:, :, 0:1]
        y_pm = y_seq[:, :, 2:3]

        for k in range(K):
            dt_k = dt_seq[:, k:k + 1]
            u_k = u_seq[:, k]
            mm_gt_prev = mm_prev
            pm_gt_prev = pm_prev

            if k == 0:
                mm_gt_prev = mm_prev
                pm_gt_prev = pm_prev

            if k % 200 == 0 and teacher_forcing:
                mm_gt_prev = y_mm[:, k - 1, :]
                pm_gt_prev = y_pm[:, k - 1, :]

            det_mm = mm_gt_prev.detach()
            det_pm = pm_gt_prev.detach()
            feat_k = torch.cat(
                [
                    torch.sqrt(u_k).clamp_min(0),
                    torch.sqrt(det_mm).clamp_min(1),
                    torch.sqrt(det_pm).clamp_min(1),
                ],
                dim=-1,
            )

            x = feat_k
            for li, cell in enumerate(self.cells):
                hs[li] = cell(x, hs[li])
                x = self.dropout(hs[li]) if self.training else hs[li]
            h = x

            z = self.bottle(h)
            raw = self.head(z)
            lam_raw, lamO_raw, VTX_mag, kdm_raw, VTL_mag, kmt_raw, kmatm_raw = raw.split(1, dim=-1)

            lam_k = gamma(lam_raw, 1e-6, 0.0005)
            lam_O = gamma(lamO_raw, 1e-6, 0.0005)
            VTXmax = gamma(VTX_mag, 5e-5, 0.1)
            VTLmax = gamma(VTL_mag, 5e-5, 0.06)
            kdm_k = gamma(kdm_raw, 1e-5, 1e-2)
            kmt_k = gamma(kmt_raw, 1e-5, 0.00035)
            kmatm_k = gamma(kmatm_raw, 5e-5, 0.0035)

            m_curr, mm_curr, p_curr, pm_curr, R_curr, O_curr = ivtt_step_R_O_mRNA_maturation(
                m_prev, mm_prev, p_prev, pm_prev, R_prev, O_prev,
                dt_k, dna_cum_total,
                lam_k, lam_O, VTXmax, kdm_k, VTLmax, kmt_k, kmatm_k,
            )

            mm_out[:, k:k + 1, :] = mm_curr.unsqueeze(1)
            p_out[:, k:k + 1, :] = p_curr.unsqueeze(1)
            pm_out[:, k:k + 1, :] = pm_curr.unsqueeze(1)

            VTX_seq[:, k:k + 1, :] = VTXmax.unsqueeze(1)
            kdm_seq[:, k:k + 1, :] = kdm_k.unsqueeze(1)
            VTL_seq[:, k:k + 1, :] = VTLmax.unsqueeze(1)
            kmt_seq[:, k:k + 1, :] = kmt_k.unsqueeze(1)
            kmatm_seq[:, k:k + 1, :] = kmatm_k.unsqueeze(1)
            R_seq[:, k:k + 1, :] = R_curr.unsqueeze(1)
            lam_seq[:, k:k + 1, :] = lam_k.unsqueeze(1)
            lamO_seq[:, k:k + 1, :] = lam_O.unsqueeze(1)

            m_prev, mm_prev, p_prev, pm_prev, R_prev, O_prev = (
                m_curr, mm_curr, p_curr, pm_curr, R_curr, O_curr
            )

        out = torch.cat([mm_out, p_out, pm_out], dim=-1)
        params = torch.cat(
            [VTX_seq, kdm_seq, VTL_seq, kmt_seq, kmatm_seq, R_seq, lam_seq, lamO_seq], dim=-1
        )
        return (out, params) if self.training else (out, params.detach())
