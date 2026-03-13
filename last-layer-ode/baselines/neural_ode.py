from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

from scaffolds import Scaffold


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
        use_basal: bool = False,
    ):
        super().__init__()
        self.U = int(U)
        self.P = int(scaffold.P)
        self.n_substeps = int(n_substeps)
        self.hidden = int(hidden)

        self.mlp = nn.Sequential(
            nn.Linear(self.U + self.P, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, self.P), # full (13 species) system output, regardless of our inputs.
        )

        # make jump move with device + saved in checkpoints
        if u_to_y_jump.shape != (self.U, self.P):
            raise ValueError(f"u_to_y_jump must be (U,P)=({self.U},{self.P}), got {tuple(u_to_y_jump.shape)}")
        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

    def rk4_Node(self, n_substeps, y: torch.Tensor, dt: torch.Tensor, u_k: torch.Tensor) -> torch.Tensor:
        # y: (B,P), dt: (B,) or (B,1), theta: (B,theta_dim), beta: (B,P) optional residual
        n_sub = max(1, int(n_substeps))

        if dt.ndim == 1:
            dt = dt.unsqueeze(1)  # (B,1)

        hdt = dt / float(n_sub)  # (B,1)

        for _ in range(n_sub):
            k1 = self.mlp(torch.cat([y, u_k], dim=-1))
            k2 = self.mlp(torch.cat([y+0.5 * hdt * k1, u_k], dim=-1))
            k3 = self.mlp(torch.cat([y+0.5 * hdt * k2, u_k], dim=-1))
            k4 = self.mlp(torch.cat([y+hdt * k3, u_k], dim=-1))
            y = y + (hdt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return torch.clamp_min(y, 0.0)
        # return y

    def forward(
        self,
        y0: torch.Tensor,                 # (B,P)
        u_seq: torch.Tensor,              # (B,K,U)
        dt_seq: torch.Tensor,             # (B,K)
        y_seq: Optional[torch.Tensor] = None,   # (B,K,P) for teacher forcing
        teacher_forcing: bool = True,
        tf_every: int = 50,
        obs_idx: Optional[torch.Tensor] = None,   # <--- ADD
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        y_out = torch.empty(B, K, self.P, device=y0.device, dtype=y0.dtype)
        th_out   = torch.zeros(B, K, 1, device=y0.device, dtype=y0.dtype) #dummy
        beta_out = torch.zeros(B, K, self.P, device=y0.device, dtype=y0.dtype) #dummy

        y_prev = y0
        
        for k in range(K):
            u_k = u_seq[:, k, :]              # (B,U)
            dt_k = dt_seq[:, k]               # (B,)

            y_in = y_prev.detach()
            if teacher_forcing and (y_seq is not None) and k > 0 and (k % tf_every == 0):
                if obs_idx is None:
                    # default: full teacher forcing (old behavior)
                    y_in = y_seq[:, k - 1, :].detach()
                else:
                    # partial teacher forcing: only overwrite observed dims
                    y_in = y_in.clone()
                    idx = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].detach()

            # jump
            y = y_in + (u_k @ self.u_to_y_jump)

            y = self.rk4_Node(1, y, dt_k, u_k)

            y_out[:, k, :] = y
            y_prev = y

        return y_out, th_out, beta_out
        


