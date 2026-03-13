from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

from scaffolds import Scaffold


def gamma(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


def rk4_substeps(rhs, n_substeps, y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    n_sub = max(1, int(n_substeps))

    if dt.ndim == 1:
        dt = dt.unsqueeze(1)

    hdt = dt / float(n_sub)
    for _ in range(n_sub):
        k1 = rhs(y, theta)
        k2 = rhs(y + 0.5 * hdt * k1, theta)
        k3 = rhs(y + 0.5 * hdt * k2, theta)
        k4 = rhs(y + hdt * k3, theta)
        y = y + (hdt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return torch.clamp_min(y, 0.0)


def shifted_context(
    y0: torch.Tensor,
    y_roll: torch.Tensor,
    y_seq: Optional[torch.Tensor] = None,
    teacher_forcing: bool = True,
    tf_every: int = 50,
    obs_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, K, P = y_roll.shape
    ctx = torch.empty(B, K, P, device=y_roll.device, dtype=y_roll.dtype)
    ctx[:, 0, :] = y0
    ctx[:, 1:, :] = y_roll[:, :-1, :]

    if teacher_forcing and (y_seq is not None):
        for k in range(1, K):
            if k % tf_every == 0:
                if obs_idx is None:
                    ctx[:, k, :] = y_seq[:, k - 1, :].detach()
                else:
                    idx = obs_idx.to(device=ctx.device, dtype=torch.long)
                    ctx[:, k, idx] = y_seq[:, k - 1, idx].detach()

    return ctx


class TrajectoryODERNN(nn.Module):
    """
    Trajectory-level closed loop:
      1. build a whole-trajectory context
      2. predict theta_1:K with a single GRU pass
      3. roll out the ODE over the whole trajectory
      4. optionally repeat
    """

    def __init__(
        self,
        *,
        U: int,
        scaffold: Scaffold,
        u_to_y_jump: torch.Tensor,
        hidden: int = 128,
        lift_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        theta_lo: float = 1e-3,
        theta_hi: float = 2.0,
        n_substeps: int = 1,
        n_passes: int = 1,
    ):
        super().__init__()
        self.U = int(U)
        self.P = int(scaffold.P)
        self.theta_dim = int(scaffold.theta_dim)
        self.rhs = scaffold.rhs
        self.n_substeps = int(n_substeps)
        self.n_passes = max(1, int(n_passes))

        self.theta_lo = float(theta_lo)
        self.theta_hi = float(theta_hi)

        self.lift = nn.Sequential(
            nn.Linear(self.U + self.P, lift_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.gru = nn.GRU(
            input_size=lift_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, self.theta_dim)

        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

    def rollout(
        self,
        y0: torch.Tensor,
        u_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        theta_seq: torch.Tensor,
    ) -> torch.Tensor:
        B, K, _ = u_seq.shape
        y_out = torch.empty(B, K, self.P, device=y0.device, dtype=y0.dtype)

        y_prev = y0
        for k in range(K):
            u_k = u_seq[:, k, :]
            dt_k = dt_seq[:, k]
            theta_k = theta_seq[:, k, :]

            y = y_prev + (u_k @ self.u_to_y_jump)
            y = rk4_substeps(self.rhs, self.n_substeps, y, dt_k, theta_k)

            y_out[:, k, :] = y
            y_prev = y

        return y_out

    def forward(
        self,
        y0: torch.Tensor,
        u_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        y_seq: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True,
        tf_every: int = 50,
        obs_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape

        y_roll = y0.unsqueeze(1).expand(-1, K, -1)
        theta_seq = torch.empty(B, K, self.theta_dim, device=y0.device, dtype=y0.dtype)

        for pass_idx in range(self.n_passes):
            ctx = shifted_context(
                y0,
                y_roll,
                y_seq=y_seq,
                teacher_forcing=teacher_forcing,
                tf_every=tf_every,
                obs_idx=obs_idx,
            )
            feat = torch.cat([u_seq, ctx], dim=-1)
            x = self.lift(feat)
            z, _ = self.gru(x)
            theta_seq = gamma(self.head(z), self.theta_lo, self.theta_hi)
            y_next = self.rollout(y0, u_seq, dt_seq, theta_seq)
            y_roll = y_next.detach() if pass_idx + 1 < self.n_passes else y_next

        return y_roll, theta_seq
