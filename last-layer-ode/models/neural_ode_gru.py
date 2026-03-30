from typing import Optional, Tuple

import torch
import torch.nn as nn


class NeuralOdeGRU(nn.Module):
    """
    Black-box GRU baseline: GRU predicts dy/dt directly, integrated with Euler.

    No mechanistic scaffold — the GRU IS the vector field. Uses Euler (not RK4)
    because the GRU hidden state cannot be rolled back; calling the RHS multiple
    times per step (as RK4 requires) would incorrectly advance the hidden state.

    Architecture:
      (u_k, y_{k-1}) -> lift (SiLU) -> GRU -> head -> dy/dt
      y_k = y_{k-1} + u_k @ jump + dt_k * dy/dt

    Returns a 3-tuple (y_out, th_out, beta_out) to match the shared interface;
    th_out and beta_out are zero-filled dummies.
    """

    def __init__(
        self,
        *,
        U: int,
        u_to_y_jump: torch.Tensor,   # (U, P)
        hidden: int = 128,
        lift_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        if u_to_y_jump.ndim != 2:
            raise ValueError(f"u_to_y_jump must be 2-D, got shape {tuple(u_to_y_jump.shape)}")
        P = int(u_to_y_jump.shape[1])
        self.U = int(U)
        self.P = P

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
        self.head = nn.Linear(hidden, self.P)

        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

    def forward(
        self,
        y0: torch.Tensor,                      # (B, P)
        u_seq: torch.Tensor,                   # (B, K, U)
        dt_seq: torch.Tensor,                  # (B, K)
        obs_idx: torch.Tensor,
        y_seq: Optional[torch.Tensor] = None,  # (B, K, P) for teacher forcing
        teacher_forcing: bool = True,
        tf_every: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        device, dtype = y0.device, y0.dtype

        y_out    = torch.empty(B, K, self.P, device=device, dtype=dtype)
        th_out   = torch.zeros(B, K, 1,      device=device, dtype=dtype)
        beta_out = torch.zeros(B, K, self.P, device=device, dtype=dtype)

        h = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size, device=device, dtype=dtype)

        use_partial = obs_idx.numel() > 0

        y_prev = y0
        for k in range(K):
            u_k  = u_seq[:, k, :]
            dt_k = dt_seq[:, k]

            y_in = y_prev.detach()
            if teacher_forcing and k > 0 and (k % tf_every == 0) and y_seq is not None:
                if use_partial:
                    y_in = y_prev.clone()
                    idx  = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].to(dtype=y_in.dtype).detach()
                else:
                    y_in = y_seq[:, k - 1, :].to(dtype=y_prev.dtype).detach()

            feat = torch.cat([u_k, y_in], dim=-1)
            x    = self.lift(feat).unsqueeze(1)
            z, h = self.gru(x, h)
            dydt = self.head(z.squeeze(1))  # (B, P)

            # Euler integration — Euler only; GRU state cannot be rolled back for RK4
            y = y_in + (u_k @ self.u_to_y_jump) + dt_k.unsqueeze(1) * dydt
            y = torch.clamp_min(y, 0.0)

            y_out[:, k, :] = y
            y_prev = y

        return y_out, th_out, beta_out
