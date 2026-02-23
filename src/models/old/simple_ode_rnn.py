from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn


def gamma(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    # bounded positive map in (lo, hi)
    return lo + (hi - lo) * torch.sigmoid(x)


def reduced_chain_rhs_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Reduced 5-state chain WITHOUT basal/source terms.

    y    : (B,5)  [A, D, G, J, M]
    theta: (B,8)  [kf1,kf2,kf3,kf4, kr1,kr2,kr3,kr4]

    returns dy/dt: (B,5)
    """
    A, D, G, J, M = y.unbind(dim=-1)
    kf1, kf2, kf3, kf4, kr1, kr2, kr3, kr4 = theta.unbind(dim=-1)

    dA = -kf1 * A + kr1 * D
    dD =  kf1 * A - kr1 * D - kf2 * D + kr2 * G
    dG =  kf2 * D - kr2 * G - kf3 * G + kr3 * J
    dJ =  kf3 * G - kr3 * J - kf4 * J + kr4 * M
    dM =  kf4 * J - kr4 * M

    return torch.stack([dA, dD, dG, dJ, dM], dim=-1)


def rk4_step(y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    One explicit RK4 step with frozen parameters theta.

    y    : (B,5)
    dt   : (B,1)
    theta: (B,8)
    """
    k1 = reduced_chain_rhs_torch(y, theta)
    k2 = reduced_chain_rhs_torch(y + 0.5 * dt * k1, theta)
    k3 = reduced_chain_rhs_torch(y + 0.5 * dt * k2, theta)
    k4 = reduced_chain_rhs_torch(y + dt * k3, theta)
    y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return torch.clamp_min(y_next, 0.0)  # concentrations can't be negative


def rk4_step_substeps(y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor, n_sub: int) -> torch.Tensor:
    """
    RK4 with substepping for stability.

    Splits dt into n_sub equal substeps (h = dt/n_sub) and applies RK4 n_sub times.
    """
    if n_sub <= 1:
        return rk4_step(y, dt, theta)

    h = dt / float(n_sub)
    y_cur = y
    for _ in range(n_sub):
        y_cur = rk4_step(y_cur, h, theta)
    return y_cur


class SimpleRNN(nn.Module):
    """
    Closed-loop kinetics learner:
      Inputs per step k: [u_k, y_prev] -> GRU -> theta_k -> RK4 step(s)

    Returns:
      out    : (B,K,P) predicted y(t1..tK)
      params : (B,K,8) predicted theta_k = [rates(8)]
    """

    def __init__(
        self,
        in_u: int,
        P: int = 5,  # number of observed species (= P_full for 5-state)
        lift_dim: int = 32,
        hidden: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        u_to_x_jump: torch.Tensor | None = None,  # shape (U, P_full)
        obs_indices: torch.Tensor | None = None,   # (n_obs,) — defaults to arange(P)
        n_sub: int = 10,  # RK4 substeps
        # bounds for parameters
        rate_lo: float = 1e-2,
        rate_hi: float = 3.0,
        # ── backward compat aliases ──
        u_to_y_jump: torch.Tensor | None = None,
        P_full: int | None = None,  # accepted for interface parity with 7/9-state, ignored
    ):
        super().__init__()
        # P_full is the unified interface name; use it when provided
        if P_full is not None:
            P = P_full
        self.P = P
        self.P_full = P  # for 5-state, latent == observed
        self.in_u = in_u
        self.hidden = hidden
        self.num_layers = num_layers
        self.n_sub = int(n_sub)

        self.rate_lo, self.rate_hi = float(rate_lo), float(rate_hi)

        # obs_indices: which positions of the latent state are observed
        if obs_indices is None:
            obs_indices = torch.arange(P, dtype=torch.long)
        self.register_buffer("obs_indices", obs_indices.to(dtype=torch.long), persistent=True)
        self.n_obs = int(obs_indices.numel())

        # Accept either name (u_to_x_jump preferred, u_to_y_jump for old code)
        jump = u_to_x_jump if u_to_x_jump is not None else u_to_y_jump
        if jump is None:
            raise ValueError("u_to_x_jump must be provided.")
        self.register_buffer("u_to_x_jump", jump.to(dtype=torch.float32), persistent=True)

        in_dim = in_u + P  # [u_k, y_prev]

        self.lift = nn.Sequential(
            nn.Linear(in_dim, lift_dim),
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

        # head -> 8 kinetic rates
        self.head = nn.Linear(hidden, 8)

        # init
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
        nn.init.constant_(self.head.bias, 0.0)

    def forward(
        self,
        y0: torch.Tensor,     # (B,P) initial observed state
        u_seq: torch.Tensor,  # (B,K,U) control inputs
        dt_seq: torch.Tensor, # (B,K) time intervals
        y_seq: torch.Tensor | None = None,  # (B,K,P) observed trajectory for teacher forcing
        teacher_forcing: bool = True,
        tf_every: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, K, U = u_seq.shape
        dev = u_seq.device
        dtype = u_seq.dtype

        y_out = torch.empty(B, K, self.P, device=dev, dtype=dtype)
        theta_out = torch.empty(B, K, 8, device=dev, dtype=dtype)

        y_prev = y0 + 0.01
        h = torch.zeros(self.num_layers, B, self.hidden, device=dev, dtype=dtype)

        for k in range(K):
            dt_k = dt_seq[:, k:k+1]  # (B,1)
            u_k  = u_seq[:, k, :]    # (B,U)

            # teacher forcing uses previous true state occasionally
            if teacher_forcing and (y_seq is not None) and k > 0 and (k % tf_every == 0):
                y_prev_in = y_seq[:, k-1, :].detach()
            else:
                y_prev_in = y_prev.detach()

            feat = torch.cat([u_k, y_prev_in], dim=-1)
            x = self.lift(feat).unsqueeze(1)     # (B,1,H)
            z, h = self.gru(x, h)                # z: (B,1,H)
            z = z.squeeze(1)                     # (B,H)

            raw = self.head(z)                   # (B,8)

            # 8 kinetic rates (positive, bounded)
            theta_k = gamma(raw, self.rate_lo, self.rate_hi)  # (B,8)

            # Apply bolus jump
            y_jump = y_prev + (u_k @ self.u_to_x_jump.to(device=dev, dtype=dtype))

            # Integrate with RK4 substepping
            y_next = rk4_step_substeps(y_jump, dt_k, theta_k, n_sub=self.n_sub)

            y_out[:, k, :] = y_next
            theta_out[:, k, :] = theta_k
            y_prev = y_next

        return (y_out, theta_out) if self.training else (y_out, theta_out.detach())