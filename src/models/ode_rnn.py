from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn


def gamma(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


class ODERNN(nn.Module):
    """
    Generic closed-loop learner:
      [u_k, y_{k-1}] -> GRU -> theta_k -> jump -> RK4 step via scaffold

    Returns:
      y_out    : (B,K,P)
      theta_out: (B,K,theta_dim)
    """

    def __init__(
        self,
        U: int,
        scaffold: nn.Module,                 # MechanisticScaffold
        hidden: int = 128,
        num_layers: int = 1,
        lift_dim: int = 32,
        dropout: float = 0.0,
        u_to_y_jump: Optional[torch.Tensor] = None,  # (U,P)
        theta_lo: float = 1e-3,
        theta_hi: float = 2.0,
        n_substeps: int = 1,                 # NEW
    ):
        super().__init__()
        self.scaffold = scaffold
        self.P = int(scaffold.spec.P)
        self.theta_dim = int(scaffold.spec.theta_dim)
        self.U = int(U)

        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.n_substeps = int(n_substeps)

        in_dim = self.U + self.P

        self.lift = nn.Sequential(
            nn.Linear(in_dim, lift_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.gru = nn.GRU(
            input_size=lift_dim,
            hidden_size=self.hidden,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )

        self.head = nn.Linear(self.hidden, self.theta_dim)

        if u_to_y_jump is None:
            raise ValueError("u_to_y_jump must be provided with shape (U,P).")
        self.register_buffer("u_to_y_jump", u_to_y_jump.to(torch.float32), persistent=True)

        self.theta_lo = float(theta_lo)
        self.theta_hi = float(theta_hi)

        # simple init
        for m in self.lift:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for name, p in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        y0: torch.Tensor,      # (B,P)
        u_seq: torch.Tensor,   # (B,K,U)
        dt_seq: torch.Tensor,  # (B,K)
        y_seq: Optional[torch.Tensor] = None,  # (B,K,P) for TF
        teacher_forcing: bool = True,
        tf_every: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, U = u_seq.shape
        dev = u_seq.device
        dtype = u_seq.dtype

        y_out = torch.empty(B, K, self.P, device=dev, dtype=dtype)
        th_out = torch.empty(B, K, self.theta_dim, device=dev, dtype=dtype)

        y_prev = y0
        h = torch.zeros(self.num_layers, B, self.hidden, device=dev, dtype=dtype)

        n_sub = max(1, self.n_substeps)

        for k in range(K):
            dt_k = dt_seq[:, k:k+1]     # (B,1)
            u_k = u_seq[:, k, :]        # (B,U)

            if teacher_forcing and (y_seq is not None) and k > 0 and (k % tf_every == 0):
                y_in = y_seq[:, k-1, :].detach()
            else:
                y_in = y_prev.detach()

            feat = torch.cat([u_k, y_in], dim=-1)          # (B,U+P)
            x = self.lift(feat).unsqueeze(1)               # (B,1,lift_dim)

            z, h = self.gru(x, h)                          # (B,1,H)
            z = z.squeeze(1)
            raw = self.head(z)                             # (B,theta_dim)

            theta_k = gamma(raw, self.theta_lo, self.theta_hi)

            y = y_prev + (u_k @ self.u_to_y_jump.to(dev, dtype))
            h_dt = dt_k / float(n_sub)

            for _ in range(n_sub):
                y = self.scaffold.step(y, h_dt, theta_k)

            y_out[:, k, :] = y
            th_out[:, k, :] = theta_k
            y_prev = y

        return (y_out, th_out) if self.training else (y_out, th_out.detach())