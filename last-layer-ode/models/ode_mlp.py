from typing import Optional, Tuple

import torch
import torch.nn as nn

from scaffolds import MechanisticScaffold


def gamma(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


class OdeMLP(nn.Module):
    """
    Memoryless MLP encoder: θ_k = MLP(u_k, y_{k-1}).

    No recurrence, no attention — purely Markovian θ prediction. Serves as the
    "no-history" ablation baseline: if this matches the GRU, temporal history
    is not needed; if it underperforms, history is load-bearing.

    Architecture:
      (U+P) -> lift_dim (SiLU) -> [hidden (SiLU)] x num_layers -> theta_dim

    Fully TorchScript-compatible (set jit_scripting: true if desired).
    """

    def __init__(
        self,
        *,
        U: int,
        rhs: MechanisticScaffold,
        u_to_y_jump: torch.Tensor,
        hidden: int = 128,
        lift_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.0,
        theta_lo: float = 1e-3,
        theta_hi: float = 2.0,
        n_substeps: int = 1,
        use_basal: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.U         = int(U)
        self.P         = int(rhs.P)
        self.theta_dim = int(rhs.theta_dim)
        self.rhs       = rhs
        self.n_substeps = int(n_substeps)
        self.use_basal  = bool(use_basal)
        self.theta_lo   = float(theta_lo)
        self.theta_hi   = float(theta_hi)

        layers: list[nn.Module] = [
            nn.Linear(self.U + self.P, lift_dim),
            nn.SiLU(),
        ]
        in_dim = lift_dim
        for _ in range(max(1, num_layers)):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        self.mlp = nn.Sequential(*layers)

        head_out = self.theta_dim + self.P if use_basal else self.theta_dim
        self.head = nn.Linear(hidden, head_out)

        if u_to_y_jump.shape != (self.U, self.P):
            raise ValueError(
                f"u_to_y_jump must be (U,P)=({self.U},{self.P}), "
                f"got {tuple(u_to_y_jump.shape)}"
            )
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

        y_out    = torch.empty(B, K, self.P,         device=device, dtype=dtype)
        th_out   = torch.empty(B, K, self.theta_dim, device=device, dtype=dtype)
        beta_out = torch.zeros(B, K, self.P,         device=device, dtype=dtype)

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

            feat = torch.cat([u_k, y_in], dim=-1)  # (B, U+P)
            z    = self.mlp(feat)                   # (B, hidden)
            raw  = self.head(z)

            if self.use_basal:
                theta_k = gamma(raw[:, :self.theta_dim], self.theta_lo, self.theta_hi)
                beta_k  = raw[:, self.theta_dim:] * (y_prev / (y_prev + 1.0))
                beta_out[:, k, :] = beta_k
                y = y_prev + (u_k @ self.u_to_y_jump)
                y = self._rk4_substeps_basal(y, dt_k, theta_k, beta_k)
            else:
                theta_k = gamma(raw, self.theta_lo, self.theta_hi)
                y = y_prev + (u_k @ self.u_to_y_jump)
                y = self._rk4_substeps(y, dt_k, theta_k)

            y_out[:, k, :]  = y
            th_out[:, k, :] = theta_k
            y_prev = y

        return y_out, th_out, beta_out

    def _rk4_substeps(
        self, y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        rhs   = self.rhs
        n_sub = self.n_substeps
        dt    = dt.unsqueeze(1)
        hdt   = dt / float(n_sub)
        for _ in range(n_sub):
            k1 = rhs(y,                   theta)
            k2 = rhs(y + 0.5 * hdt * k1, theta)
            k3 = rhs(y + 0.5 * hdt * k2, theta)
            k4 = rhs(y +       hdt * k3,  theta)
            y  = y + (hdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.clamp_min(y, 0.0)

    def _rk4_substeps_basal(
        self,
        y: torch.Tensor,
        dt: torch.Tensor,
        theta: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        rhs   = self.rhs
        n_sub = self.n_substeps
        dt    = dt.unsqueeze(1)
        hdt   = dt / float(n_sub)
        for _ in range(n_sub):
            k1 = rhs(y,                   theta) + beta
            k2 = rhs(y + 0.5 * hdt * k1, theta) + beta
            k3 = rhs(y + 0.5 * hdt * k2, theta) + beta
            k4 = rhs(y +       hdt * k3,  theta) + beta
            y  = y + (hdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.clamp_min(y, 0.0)
