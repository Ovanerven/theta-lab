from typing import Optional, Tuple

import torch
import torch.nn as nn

from scaffolds import MechanisticScaffold


def gamma(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


class OdeRNN2020(nn.Module):
    """
    ODE-RNN style model inspired by the 2020 formulation:
      1) integrate latent hidden state with an ODE between observations
      2) update latent state with a recurrent unit at each observation step
      3) decode latent state to mechanistic theta_k
      4) integrate mechanistic scaffold ODE for y_k over dt_k

    Keeps the same training pipeline interface as other models:
      forward(...) -> (y_out, th_out, beta_out)
    """

    def __init__(
        self,
        *,
        U: int,
        rhs: MechanisticScaffold,
        u_to_y_jump: torch.Tensor,   # (U,P)
        hidden: int = 128,
        lift_dim: int = 32,
        num_layers: int = 1,         # kept for API compatibility
        dropout: float = 0.0,
        theta_lo: float = 1e-3,
        theta_hi: float = 2.0,
        n_substeps: int = 1,
        use_basal: bool = False,
        ode_latent_layers: int = 2,
        ode_latent_substeps: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.U = int(U)
        self.P = int(rhs.P)
        self.theta_dim = int(rhs.theta_dim)
        self.rhs = rhs
        self.n_substeps = int(n_substeps)
        self.use_basal = bool(use_basal)
        self.theta_lo = float(theta_lo)
        self.theta_hi = float(theta_hi)
        self.ode_latent_substeps = int(max(1, ode_latent_substeps))

        self.input_proj = nn.Sequential(
            nn.Linear(self.U + self.P, lift_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(lift_dim, hidden),
            nn.SiLU(),
        )

        latent_layers = []
        in_dim = hidden
        n_latent_layers = int(max(1, ode_latent_layers))
        for layer_idx in range(n_latent_layers - 1):
            latent_layers.append(nn.Linear(in_dim, hidden))
            latent_layers.append(nn.Tanh())
            in_dim = hidden
        latent_layers.append(nn.Linear(in_dim, hidden))
        self.hidden_ode_func = nn.Sequential(*latent_layers)

        self.gru_cell = nn.GRUCell(input_size=hidden, hidden_size=hidden)

        head_out = self.theta_dim + self.P if self.use_basal else self.theta_dim
        self.head = nn.Linear(hidden, head_out)

        if u_to_y_jump.shape != (self.U, self.P):
            raise ValueError(
                f"u_to_y_jump must be (U,P)=({self.U},{self.P}), got {tuple(u_to_y_jump.shape)}"
            )
        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

    def _rk4_hidden(self, h: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        dt = dt.unsqueeze(1)
        hdt = dt / float(self.ode_latent_substeps)
        for _ in range(self.ode_latent_substeps):
            k1 = self.hidden_ode_func(h)
            k2 = self.hidden_ode_func(h + 0.5 * hdt * k1)
            k3 = self.hidden_ode_func(h + 0.5 * hdt * k2)
            k4 = self.hidden_ode_func(h + hdt * k3)
            h = h + (hdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return h

    def _rk4_substeps(
        self,
        y: torch.Tensor,
        dt: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        rhs = self.rhs
        n_sub = self.n_substeps
        dt = dt.unsqueeze(1)
        hdt = dt / float(n_sub)
        for _ in range(n_sub):
            k1 = rhs(y,                   theta)
            k2 = rhs(y + 0.5 * hdt * k1, theta)
            k3 = rhs(y + 0.5 * hdt * k2, theta)
            k4 = rhs(y +       hdt * k3, theta)
            y  = y + (hdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.clamp_min(y, 0.0)

    def _rk4_substeps_basal(
        self,
        y: torch.Tensor,
        dt: torch.Tensor,
        theta: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        rhs = self.rhs
        n_sub = self.n_substeps
        dt = dt.unsqueeze(1)
        hdt = dt / float(n_sub)
        for _ in range(n_sub):
            k1 = rhs(y,                   theta) + beta
            k2 = rhs(y + 0.5 * hdt * k1, theta) + beta
            k3 = rhs(y + 0.5 * hdt * k2, theta) + beta
            k4 = rhs(y +       hdt * k3, theta) + beta
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
        y_out = torch.empty(B, K, self.P, device=y0.device, dtype=y0.dtype)
        th_out = torch.empty(B, K, self.theta_dim, device=y0.device, dtype=y0.dtype)
        beta_out = torch.zeros(B, K, self.P, device=y0.device, dtype=y0.dtype)

        h = torch.zeros(B, self.gru_cell.hidden_size, device=y0.device, dtype=y0.dtype)

        use_partial = obs_idx.numel() > 0
        y_prev = y0

        for k in range(K):
            u_k = u_seq[:, k, :]
            dt_k = dt_seq[:, k]

            # latent ODE evolution between time points
            h_ode = self._rk4_hidden(h, dt_k)

            y_in = y_prev.detach()
            if teacher_forcing and k > 0 and (k % tf_every == 0) and y_seq is not None:
                if use_partial:
                    y_in = y_prev.clone()
                    idx = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].to(dtype=y_in.dtype).detach()
                else:
                    y_in = y_seq[:, k - 1, :].to(dtype=y_prev.dtype).detach()

            x_in = torch.cat([u_k, y_in], dim=-1)
            x_proj = self.input_proj(x_in)

            # ODE-RNN update (RNN cell receives ODE-evolved hidden state)
            h = self.gru_cell(x_proj, h_ode)
            raw = self.head(h)

            if self.use_basal:
                theta_k = gamma(raw[:, :self.theta_dim], self.theta_lo, self.theta_hi)
                beta_k = raw[:, self.theta_dim:] * (y_prev / (y_prev + 1.0))
                beta_out[:, k, :] = beta_k
                y = y_prev + (u_k @ self.u_to_y_jump)
                y = self._rk4_substeps_basal(y, dt_k, theta_k, beta_k)
            else:
                theta_k = gamma(raw, self.theta_lo, self.theta_hi)
                y = y_prev + (u_k @ self.u_to_y_jump)
                y = self._rk4_substeps(y, dt_k, theta_k)

            y_out[:, k, :] = y
            th_out[:, k, :] = theta_k
            y_prev = y

        return y_out, th_out, beta_out
