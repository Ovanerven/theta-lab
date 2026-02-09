from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def gamma(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    # bounded positive map in (lo, hi)
    return lo + (hi - lo) * torch.sigmoid(x)


def reduced_chain_rhs_torch(y: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Torch equivalent of benchmark_models.ReducedModel
    y: (B,5) [A, D, G, J, M]
    k: (B,8) [kf1,kf2,kf3,kf4, kr1,kr2,kr3,kr4]
    returns dy/dt: (B,5)
    """
    A, D, G, J, M = y.unbind(dim=-1)
    kf1, kf2, kf3, kf4, kr1, kr2, kr3, kr4 = k.unbind(dim=-1)

    dA = -kf1 * A + kr1 * D
    dD =  kf1 * A - kr1 * D - kf2 * D + kr2 * G
    dG =  kf2 * D - kr2 * G - kf3 * G + kr3 * J
    dJ =  kf3 * G - kr3 * J - kf4 * J + kr4 * M
    dM =  kf4 * J - kr4 * M

    return torch.stack([dA, dD, dG, dJ, dM], dim=-1)


def rk4_step(y: torch.Tensor, dt: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    One explicit RK4 step with frozen parameters k.
    y : (B,5)
    dt: (B,1)
    k : (B,8)
    """
    k1 = reduced_chain_rhs_torch(y, k)
    k2 = reduced_chain_rhs_torch(y + 0.5 * dt * k1, k)
    k3 = reduced_chain_rhs_torch(y + 0.5 * dt * k2, k)
    k4 = reduced_chain_rhs_torch(y + dt * k3, k)
    y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return torch.clamp_min(y_next, 0.0) # y can't be negative


class SimpleRNN(nn.Module):
    """
    Closed-loop kinetics learner:
      Inputs per step k: [u_k, y_prev] -> GRU -> theta_k -> RK4 step

    Returns:
      out    : (B,K,P) predicted y(t1..tK)
      params : (B,K,8) predicted theta_k
    """

    def __init__(
        self,
        in_u: int,
        P: int = 5, # number of observed species
        lift_dim: int = 32,
        hidden: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        u_to_y_jump: torch.Tensor | None = None,  # shape (U,P)
    ):
        super().__init__()
        self.P = P # number of observed species
        self.in_u = in_u
        self.lift_dim = lift_dim
        self.hidden = hidden
        self.num_layers = num_layers

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

        # head -> 8 reduced rates
        self.head = nn.Linear(hidden, 8)

        # Jump map: converts bolus inputs u_k into an increment for the reduced state.
        # If you inject A only, this is [[1,0,0,0,0]].
        if u_to_y_jump is None:
            raise ValueError(
                "u_to_y_jump must be provided. Build it from dataset control_indices and obs_indices."
            )
        self.register_buffer("u_to_y_jump", u_to_y_jump.to(dtype=torch.float32), persistent=True)

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
        nn.init.constant_(self.head.bias, 0.0)  # start with rates near middle of gamma range

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
        # y_prev = y0

        h = torch.zeros(self.num_layers, B, self.hidden, device=dev, dtype=dtype)

        for k in range(K):
            dt_k = dt_seq[:, k:k+1]  # (B,1)
            u_k  = u_seq[:, k, :]    # (B,U)

            # teacher forcing uses previous true state occasionally
            if teacher_forcing and (y_seq is not None) and k > 0 and (k % tf_every == 0):
                y_prev_in = y_seq[:, k-1, :].detach()
            else:
                y_prev_in = y_prev.detach()

            # concatenate inputs directly
            feat = torch.cat([u_k, y_prev_in], dim=-1)
            x = self.lift(feat).unsqueeze(1)            # (B,1,H)

            # GRU update
            z, h = self.gru(x, h)                       # z: (B,1,H)
            z = z.squeeze(1)                            # (B,H)

            raw = self.head(z)                          # (B,8)

            # bounded rates
            kf1 = gamma(raw[:, 0:1], 1e-3, 2.0)
            kf2 = gamma(raw[:, 1:2], 1e-3, 2.0)
            kf3 = gamma(raw[:, 2:3], 1e-3, 2.0)
            kf4 = gamma(raw[:, 3:4], 1e-3, 2.0)
            kr1 = gamma(raw[:, 4:5], 1e-3, 2.0)
            kr2 = gamma(raw[:, 5:6], 1e-3, 2.0)
            kr3 = gamma(raw[:, 6:7], 1e-3, 2.0)
            kr4 = gamma(raw[:, 7:8], 1e-3, 2.0)
            theta_k = torch.cat([kf1, kf2, kf3, kf4, kr1, kr2, kr3, kr4], dim=-1)  # (B,8)
            
            # Apply bolus jump with physically reasonable upper bound
            # (prevents RK4 overflow while model learns appropriate parameter regime)
            y_jump = y_prev + (u_k @ self.u_to_y_jump.to(device=dev, dtype=dtype))
            # y_jump = torch.clamp(y_jump, min=0.0, max=50.0)
            y_next = rk4_step(y_jump, dt_k, theta_k)

            y_out[:, k, :] = y_next
            theta_out[:, k, :] = theta_k

            y_prev = y_next

        return (y_out, theta_out) if self.training else (y_out, theta_out.detach())
