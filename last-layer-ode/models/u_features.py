"""Shared encoder u-feature transforms.

Single source of truth for the channel-EXPANDING u_transforms so every encoder
(GRU/LSTM/sLSTM/minGRU/LMU/Mamba/transformer) produces byte-identical features.
Logic mirrors OdeRNN exactly:
  - channel-expanding modes select gru_u_cols FIRST, then stack channels (so the
    returned tensor already has cols applied — the per-step loop must NOT re-select);
  - non-expanding modes transform then select cols.

TorchScript-compatible (plain ops + typed args), so jit-scripted encoders can call it.
"""
from typing import List

import torch


def u_feature_mult(u_transform: str) -> int:
    """Per-u-column channel multiplier for a given transform."""
    if u_transform == "pulse_cumsum_sqrt" or u_transform == "cumsum_timesince_sqrt":
        return 2
    return 1


def _time_since(u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Per-channel elapsed time since the last nonzero dose (log1p-compressed)."""
    B = u.shape[0]
    K = u.shape[1]
    C = u.shape[2]
    out = torch.zeros(B, K, C, device=u.device, dtype=u.dtype)
    tsince = torch.zeros(B, C, device=u.device, dtype=u.dtype)
    event = u.abs() > 1e-8
    for k in range(K):
        tsince = tsince + dt[:, k].unsqueeze(1)
        tsince = torch.where(event[:, k, :], torch.zeros_like(tsince), tsince)
        out[:, k, :] = tsince
    return torch.log1p(out)


def build_u_enc(
    u_seq: torch.Tensor,        # (B,K,U) raw deltas
    dt_seq: torch.Tensor,       # (B,K)
    u_transform: str,
    gru_u_idx: torch.Tensor,    # long indices of u columns to keep ([] when has_u_cols=False)
    has_u_cols: bool,
) -> torch.Tensor:
    """Encoder's view of u with gru_u_cols ALREADY applied.

    Returns (B, K, C*mult) where C = #kept cols. The ODE bolus jump must still use
    the raw u_seq separately.
    """
    # Channel-EXPANDING: select cols first, then expand.
    if u_transform == "pulse_cumsum_sqrt" or u_transform == "cumsum_timesince_sqrt":
        u_base = torch.index_select(u_seq, 2, gru_u_idx) if has_u_cols else u_seq
        if u_transform == "pulse_cumsum_sqrt":
            pulse = u_base.clamp_min(1e-6).sqrt()                 # exact event timing+magnitude
            cum = u_base.cumsum(dim=1).clamp_min(1e-6).sqrt()     # persistent running recipe
            return torch.cat([pulse, cum], dim=2)
        else:  # cumsum_timesince_sqrt
            cum = u_base.cumsum(dim=1).clamp_min(1e-6).sqrt()     # magnitude
            tsince = _time_since(u_base, dt_seq)                  # recency
            return torch.cat([cum, tsince], dim=2)

    # Non-expanding: transform, then select cols.
    if u_transform == "cumsum" or u_transform == "cumsum_sqrt":
        u_enc = u_seq.cumsum(dim=1)
    else:
        u_enc = u_seq
    if u_transform == "sqrt" or u_transform == "cumsum_sqrt":
        u_enc = u_enc.clamp_min(1e-6).sqrt()
    if has_u_cols:
        u_enc = torch.index_select(u_enc, 2, gru_u_idx)
    return u_enc
