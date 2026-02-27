from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

from scaffolds import Scaffold


def gamma(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


def rk4_substeps(rhs, n_substeps, y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    # y: (B,P), dt: (B,) or (B,1), theta: (B,theta_dim)
    n_sub = max(1, int(n_substeps))

    if dt.ndim == 1:
        dt = dt.unsqueeze(1)  # (B,1)

    hdt = dt / float(n_sub)  # (B,1)

    for _ in range(n_sub):
        k1 = rhs(y, theta)
        k2 = rhs(y + 0.5 * hdt * k1, theta)
        k3 = rhs(y + 0.5 * hdt * k2, theta)
        k4 = rhs(y + hdt * k3, theta)
        y = y + (hdt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return torch.clamp_min(y, 0.0)

# We can apply the jump smoothly across the substepped integrator, which is more stable for large jumps and allows us to use fewer substeps.
# def rk4_substeps(rhs, n_substeps, y, dt, theta, *, u_k=None, jump=None):
#     n_sub = max(1, int(n_substeps))
#     if dt.ndim == 1: dt = dt.unsqueeze(1)
#     hdt = dt / float(n_sub)

#     if u_k is not None:
#         du = (u_k @ jump) / float(n_sub)   # (B,P)
#     else:
#         du = None

#     for _ in range(n_sub):
#         if du is not None:
#             y = y + du
#         k1 = rhs(y, theta)
#         k2 = rhs(y + 0.5 * hdt * k1, theta)
#         k3 = rhs(y + 0.5 * hdt * k2, theta)
#         k4 = rhs(y + hdt * k3, theta)
#         y = y + (hdt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

#     return torch.clamp_min(y, 0.0)

# Or we can just apply the jump once at the midpoint, which might be more realistic.
# def rk4_substeps(rhs, n_substeps, y, dt, theta, *, u_k=None, jump=None):
#     n_sub = max(1, int(n_substeps))
#     if dt.ndim == 1:
#         dt = dt.unsqueeze(1)  # (B,1)
#     hdt = dt / float(n_sub)   # (B,1)

#     # total jump for this coarse step (apply once, at midpoint)
#     if u_k is not None:
#         if jump is None:
#             raise ValueError("need jump if u_k provided")
#         du_total = (u_k @ jump)  # (B,P)
#         mid = n_sub // 2
#     else:
#         du_total = None
#         mid = None

#     for s in range(n_sub):
#         if du_total is not None and s == mid:
#             y = y + du_total

#         k1 = rhs(y, theta)
#         k2 = rhs(y + 0.5 * hdt * k1, theta)
#         k3 = rhs(y + 0.5 * hdt * k2, theta)
#         k4 = rhs(y + hdt * k3, theta)
#         y = y + (hdt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

#     return torch.clamp_min(y, 0.0)


class ODERNN(nn.Module):
    """
    Closed-loop:
      (u_k, y_{k-1}) -> GRU -> theta_k
      y <- y + u_k @ jump
      y <- integrate ODE with theta_k over dt_k
    """

    def __init__(
        self,
        *,
        U: int,
        scaffold: Scaffold,
        u_to_y_jump: torch.Tensor,   # (U,P)
        hidden: int = 128,
        lift_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        theta_lo: float = 1e-3,
        theta_hi: float = 2.0,
        n_substeps: int = 1,
    ):
        super().__init__()
        self.U = int(U)
        self.P = int(scaffold.P)
        self.theta_dim = int(scaffold.theta_dim)
        self.rhs = scaffold.rhs
        self.n_substeps = int(n_substeps)

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

        # make jump move with device + saved in checkpoints
        if u_to_y_jump.shape != (self.U, self.P):
            raise ValueError(f"u_to_y_jump must be (U,P)=({self.U},{self.P}), got {tuple(u_to_y_jump.shape)}")
        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

    def forward(
        self,
        y0: torch.Tensor,                 # (B,P)
        u_seq: torch.Tensor,              # (B,K,U)
        dt_seq: torch.Tensor,             # (B,K)
        y_seq: Optional[torch.Tensor] = None,   # (B,K,P) for teacher forcing
        teacher_forcing: bool = True,
        tf_every: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        y_out = torch.empty(B, K, self.P, device=y0.device, dtype=y0.dtype)
        th_out = torch.empty(B, K, self.theta_dim, device=y0.device, dtype=y0.dtype)

        h = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size, device=y0.device, dtype=y0.dtype)

        y_prev = y0
        for k in range(K):
            u_k = u_seq[:, k, :]              # (B,U)
            dt_k = dt_seq[:, k]               # (B,)

            if teacher_forcing and (y_seq is not None) and k > 0 and (k % tf_every == 0):
                y_in = y_seq[:, k - 1, :].detach()
            else:
                y_in = y_prev.detach()

            feat = torch.cat([u_k, y_in], dim=-1)      # (B,U+P)
            x = self.lift(feat).unsqueeze(1)           # (B,1,lift_dim)
            z, h = self.gru(x, h)
            raw = self.head(z.squeeze(1))              # (B,theta_dim)
            theta_k = gamma(raw, self.theta_lo, self.theta_hi)

            # jump
            y = y_prev + (u_k @ self.u_to_y_jump)

            y = rk4_substeps(self.rhs, self.n_substeps, y, dt_k, theta_k)

            # We can jump within the rk4 substepper to increase the integration resolution at a bolus DIDNT WORK. 
            # y = rk4_substeps(self.rhs, self.n_substeps, y_prev, dt_k, theta_k, u_k=u_k, jump=self.u_to_y_jump)

            y_out[:, k, :] = y
            th_out[:, k, :] = theta_k
            y_prev = y

        return y_out, th_out