from typing import Optional, Tuple

import torch
import torch.nn as nn


class NeuralODE(nn.Module):
    """
    Baseline: plain MLP parameterises the full vector field directly.
    No mechanistic scaffold — the MLP IS the rhs.
    Returns a 3-tuple (y_out, th_out, beta_out) to match the shared interface;
    th_out and beta_out are zero-filled dummies.
    """

    def __init__(
        self,
        *,
        U: int,
        u_to_y_jump: torch.Tensor,   # (U,P)
        hidden: int = 128,
        lift_dim: int = 32,          # unused, kept for API compatibility
        num_layers: int = 1,         # unused, kept for API compatibility
        dropout: float = 0.0,
        theta_lo: float = 1e-3,      # unused, kept for API compatibility
        theta_hi: float = 2.0,       # unused, kept for API compatibility
        n_substeps: int = 1,
        use_basal: bool = False,     # unused, kept for API compatibility
        **kwargs,
    ):
        super().__init__()
        if u_to_y_jump.ndim != 2:
            raise ValueError(f"u_to_y_jump must be 2-D, got shape {tuple(u_to_y_jump.shape)}")
        P = int(u_to_y_jump.shape[1])
        self.U = int(U)
        self.P = P
        self.n_substeps = int(n_substeps)

        self.mlp = nn.Sequential(
            nn.Linear(self.U + self.P, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.P),
        )

        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

    def _rk4_substeps(
        self,
        y: torch.Tensor,
        dt: torch.Tensor,
        u_k: torch.Tensor,
    ) -> torch.Tensor:
        n_sub = self.n_substeps
        hdt = dt.unsqueeze(1) / float(n_sub)
        for _ in range(n_sub):
            k1 = self.mlp(torch.cat([y,                   u_k], dim=-1))
            k2 = self.mlp(torch.cat([y + 0.5 * hdt * k1, u_k], dim=-1))
            k3 = self.mlp(torch.cat([y + 0.5 * hdt * k2, u_k], dim=-1))
            k4 = self.mlp(torch.cat([y +       hdt * k3,  u_k], dim=-1))
            y  = y + (hdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.clamp_min(y, 0.0)

    def forward(
        self,
        y0: torch.Tensor,                     # (B,P)
        u_seq: torch.Tensor,                  # (B,K,U)
        dt_seq: torch.Tensor,                 # (B,K)
        obs_idx: torch.Tensor,                # (num_obs,) — pass torch.arange(P) for full TF
        y_seq: Optional[torch.Tensor] = None, # (B,K,P) for teacher forcing
        teacher_forcing: bool = True,
        tf_every: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        y_out    = torch.empty(B, K, self.P, device=y0.device, dtype=y0.dtype)
        th_out   = torch.zeros(B, K, 1,      device=y0.device, dtype=y0.dtype)
        beta_out = torch.zeros(B, K, self.P, device=y0.device, dtype=y0.dtype)

        use_partial = obs_idx.numel() > 0
        has_y_seq   = y_seq is not None

        y_prev = y0
        for k in range(K):
            u_k  = u_seq[:, k, :]
            dt_k = dt_seq[:, k]

            y_in = y_prev.detach()

            if teacher_forcing and k > 0 and (k % tf_every == 0) and has_y_seq:
                if use_partial:
                    y_in = y_prev.clone()
                    idx  = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].to(dtype=y_in.dtype).detach()  # type: ignore[index]
                else:
                    y_in = y_seq[:, k - 1, :].to(dtype=y_prev.dtype).detach()  # type: ignore[index]

            y = y_in + (u_k @ self.u_to_y_jump)
            y = self._rk4_substeps(y, dt_k, u_k)

            y_out[:, k, :] = y
            y_prev = y

        return y_out, th_out, beta_out
