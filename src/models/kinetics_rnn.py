from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.sim.mechanisms import (
    full13_rhs, reduced5_rhs, reduced7_rhs, reduced8_rhs, reduced9_rhs,
)


def gamma_sigmoid(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


# ---------------------------------------------------------------------------
# Concrete RK4 integrators per mechanism — required for torch.jit.script
# (TorchScript cannot call arbitrary Python callables, so each mechanism
#  needs its own inlined loop.)
# ---------------------------------------------------------------------------

def rk4_full13(y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor, n_sub: int) -> torch.Tensor:
    h = dt / float(n_sub)
    for _ in range(n_sub):
        k1 = full13_rhs(y, theta)
        k2 = full13_rhs(y + 0.5 * h * k1, theta)
        k3 = full13_rhs(y + 0.5 * h * k2, theta)
        k4 = full13_rhs(y + h * k3, theta)
        y = torch.clamp_min(y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4), 0.0)
    return y


def rk4_reduced5(y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor, n_sub: int) -> torch.Tensor:
    h = dt / float(n_sub)
    for _ in range(n_sub):
        k1 = reduced5_rhs(y, theta)
        k2 = reduced5_rhs(y + 0.5 * h * k1, theta)
        k3 = reduced5_rhs(y + 0.5 * h * k2, theta)
        k4 = reduced5_rhs(y + h * k3, theta)
        y = torch.clamp_min(y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4), 0.0)
    return y


def rk4_reduced7(y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor, n_sub: int) -> torch.Tensor:
    h = dt / float(n_sub)
    for _ in range(n_sub):
        k1 = reduced7_rhs(y, theta)
        k2 = reduced7_rhs(y + 0.5 * h * k1, theta)
        k3 = reduced7_rhs(y + 0.5 * h * k2, theta)
        k4 = reduced7_rhs(y + h * k3, theta)
        y = torch.clamp_min(y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4), 0.0)
    return y


def rk4_reduced8(y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor, n_sub: int) -> torch.Tensor:
    h = dt / float(n_sub)
    for _ in range(n_sub):
        k1 = reduced8_rhs(y, theta)
        k2 = reduced8_rhs(y + 0.5 * h * k1, theta)
        k3 = reduced8_rhs(y + 0.5 * h * k2, theta)
        k4 = reduced8_rhs(y + h * k3, theta)
        y = torch.clamp_min(y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4), 0.0)
    return y


def rk4_reduced9(y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor, n_sub: int) -> torch.Tensor:
    h = dt / float(n_sub)
    for _ in range(n_sub):
        k1 = reduced9_rhs(y, theta)
        k2 = reduced9_rhs(y + 0.5 * h * k1, theta)
        k3 = reduced9_rhs(y + 0.5 * h * k2, theta)
        k4 = reduced9_rhs(y + h * k3, theta)
        y = torch.clamp_min(y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4), 0.0)
    return y


class KineticsRNN(nn.Module):
    def __init__(
        self,
        mech,                            # Mechanism from MECH dict
        obs_indices: List[int],          # which full-state species are observed
        in_u: int,
        u_to_y_jump: torch.Tensor,       # (U, P): maps control -> observed species
        lift_dim: int = 32,
        hidden: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        n_sub: int = 1,
    ):
        super().__init__()
        self.mech_name: str = mech.name   # stored as str for TorchScript dispatch
        self.obs_indices = obs_indices     # metadata for checkpointing / plotting
        self.P: int = len(obs_indices)
        self.n_state: int = mech.n_state
        self.n_param: int = mech.n_param
        self.n_sub = n_sub

        self.lift = nn.Sequential(
            nn.Linear(in_u + self.P, lift_dim),
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
        self.head = nn.Linear(hidden, mech.n_param)

        self.register_buffer("u_to_y_jump", u_to_y_jump.to(torch.float32))
        self.register_buffer("obs_idx_buf", torch.tensor(obs_indices, dtype=torch.long))

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
        y0: torch.Tensor,                          # (B, P)
        u_seq: torch.Tensor,                       # (B, K, U)
        dt_seq: torch.Tensor,                      # (B, K)
        y_seq: Optional[torch.Tensor] = None,      # (B, K, P) for teacher forcing
        teacher_forcing: bool = True,
        tf_every: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, K, U = u_seq.shape
        dev = u_seq.device
        dtype = u_seq.dtype

        obs_idx = self.obs_idx_buf.to(device=dev)  # actual positions in n_state vector
        y_hat     = torch.empty(B, K, self.P,       device=dev, dtype=dtype)
        theta_out = torch.empty(B, K, self.n_param, device=dev, dtype=dtype)

        y_full = torch.zeros(B, self.n_state, device=dev, dtype=dtype)
        y_full[:, obs_idx] = y0

        h = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size, device=dev, dtype=dtype)

        for k in range(K):
            u_k  = u_seq[:, k, :]      # (B, U)
            dt_k = dt_seq[:, k:k+1]   # (B, 1)
            y_obs = y_full[:, obs_idx] # (B, P)

            if teacher_forcing and (y_seq is not None) and k > 0 and (k % tf_every == 0):
                y_in = y_seq[:, k-1, :].detach()
            else:
                y_in = y_obs.detach()

            feat = torch.cat([u_k, y_in], dim=-1)
            x, h = self.gru(self.lift(feat).unsqueeze(1), h)
            theta_k = gamma_sigmoid(self.head(x.squeeze(1)), 1e-3, 2.0)

            # apply bolus to observed species
            y_full_jump = y_full.clone()
            y_full_jump[:, obs_idx] = y_obs + (u_k @ self.u_to_y_jump.to(dtype=dtype))

            # integrate — dispatch by mech_name (required for TorchScript)
            if self.mech_name == "full13":
                y_full = rk4_full13(y_full_jump, dt_k, theta_k, self.n_sub)
            elif self.mech_name == "reduced7":
                y_full = rk4_reduced7(y_full_jump, dt_k, theta_k, self.n_sub)
            elif self.mech_name == "reduced8":
                y_full = rk4_reduced8(y_full_jump, dt_k, theta_k, self.n_sub)
            elif self.mech_name == "reduced9":
                y_full = rk4_reduced9(y_full_jump, dt_k, theta_k, self.n_sub)
            else:
                y_full = rk4_reduced5(y_full_jump, dt_k, theta_k, self.n_sub)

            y_hat[:, k, :]     = y_full[:, obs_idx]
            theta_out[:, k, :] = theta_k

        return (y_hat, theta_out) if self.training else (y_hat, theta_out.detach())
