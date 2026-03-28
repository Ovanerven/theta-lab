from typing import Optional, Tuple

import torch
import torch.nn as nn

from scaffolds import MechanisticScaffold


def gamma(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


class ODERNNAnalytical(nn.Module):
    """
    Same closed-loop architecture as ODERNN, but with an exact linear propagation step.

    This is exact for scaffolds whose RHS is homogeneous linear in the state and whose
    theta_k is held constant over each interval [t_k, t_{k+1}], which matches the current
    chain scaffold family in scaffolds.py.
    """

    def __init__(
        self,
        *,
        U: int,
        rhs: MechanisticScaffold,
        u_to_y_jump: torch.Tensor,
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
        self.P = int(rhs.P)
        self.theta_dim = int(rhs.theta_dim)
        self.rhs = rhs

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

        if u_to_y_jump.shape != (self.U, self.P):
            raise ValueError(f"u_to_y_jump must be (U,P)=({self.U},{self.P}), got {tuple(u_to_y_jump.shape)}")
        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)
        self.register_buffer("_basis", torch.eye(self.P, dtype=torch.float32), persistent=False)

    def forward(
        self,
        y0: torch.Tensor,
        u_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        obs_idx: torch.Tensor,
        y_seq: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True,
        tf_every: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, steps, _ = u_seq.shape
        y_out = torch.empty(batch_size, steps, self.P, device=y0.device, dtype=y0.dtype)
        th_out = torch.empty(batch_size, steps, self.theta_dim, device=y0.device, dtype=y0.dtype)

        h = torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_size, device=y0.device, dtype=y0.dtype)
        basis = self._basis.to(device=y0.device, dtype=y0.dtype)

        y_prev = y0
        for k in range(steps):
            u_k = u_seq[:, k, :]
            dt_k = dt_seq[:, k]

            y_in = y_prev.detach()

            if teacher_forcing and k > 0 and (k % tf_every == 0) and y_seq is not None:
                if obs_idx.numel() > 0:
                    y_in = y_prev.clone()
                    idx = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].to(dtype=y_in.dtype).detach()
                else:
                    y_in = y_seq[:, k - 1, :].to(dtype=y_prev.dtype).detach()

            feat = torch.cat([u_k, y_in], dim=-1)
            x = self.lift(feat).unsqueeze(1)
            z, h = self.gru(x, h)
            raw = self.head(z.squeeze(1))
            theta_k = gamma(raw, self.theta_lo, self.theta_hi)

            y = y_prev + (u_k @ self.u_to_y_jump)
            y = self._exact_linear_step(basis, y, dt_k, theta_k)

            y_out[:, k, :] = y
            th_out[:, k, :] = theta_k
            y_prev = y

        return y_out, th_out

    def _linear_system_matrix(self, basis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Build batch system matrix A(theta) for homogeneous linear ODE y' = A y."""
        batch_size = theta.shape[0]
        state_dim = basis.shape[0]
        theta_dim = theta.shape[1]

        basis_batch = basis.unsqueeze(0).expand(batch_size, state_dim, state_dim)
        y_basis = basis_batch.reshape(batch_size * state_dim, state_dim)
        theta_rep = theta.unsqueeze(1).expand(batch_size, state_dim, theta_dim)
        theta_rep = theta_rep.reshape(batch_size * state_dim, theta_dim)

        cols = self.rhs(y_basis, theta_rep).reshape(batch_size, state_dim, state_dim)
        return cols.transpose(1, 2)

    def _exact_linear_step(
        self,
        basis: torch.Tensor,
        y: torch.Tensor,
        dt: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        if dt.ndim == 1:
            dt = dt.unsqueeze(1)

        system = self._linear_system_matrix(basis, theta)
        propagator = torch.matrix_exp(system * dt.unsqueeze(-1))
        y_next = torch.bmm(propagator, y.unsqueeze(-1)).squeeze(-1)
        return torch.clamp_min(y_next, 0.0)


ODERNN = ODERNNAnalytical