from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn


def gamma(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Bounded positive map in (lo, hi)."""
    return lo + (hi - lo) * torch.sigmoid(x)


# ---------------------------
# ODE (NO basal terms anymore)
# ---------------------------

def reduced_chain_rhs_torch(y: torch.Tensor, rates: torch.Tensor) -> torch.Tensor:
    """
    Reduced 5-state chain dynamics (A -> D -> G -> J -> M) with reversible steps.

    y     : (B,5)  [A, D, G, J, M]
    rates : (B,8)  [kf1,kf2,kf3,kf4, kr1,kr2,kr3,kr4]

    returns dy/dt: (B,5)
    """
    A, D, G, J, M = y.unbind(dim=-1)
    kf1, kf2, kf3, kf4, kr1, kr2, kr3, kr4 = rates.unbind(dim=-1)

    dA = -kf1 * A + kr1 * D
    dD =  kf1 * A - kr1 * D - kf2 * D + kr2 * G
    dG =  kf2 * D - kr2 * G - kf3 * G + kr3 * J
    dJ =  kf3 * G - kr3 * J - kf4 * J + kr4 * M
    dM =  kf4 * J - kr4 * M

    return torch.stack([dA, dD, dG, dJ, dM], dim=-1)


def rk4_step(y: torch.Tensor, dt: torch.Tensor, rates: torch.Tensor) -> torch.Tensor:
    """
    One explicit RK4 step with frozen parameters rates.

    y     : (B,5)
    dt    : (B,1)
    rates : (B,8)
    """
    k1 = reduced_chain_rhs_torch(y, rates)
    k2 = reduced_chain_rhs_torch(y + 0.5 * dt * k1, rates)
    k3 = reduced_chain_rhs_torch(y + 0.5 * dt * k2, rates)
    k4 = reduced_chain_rhs_torch(y + dt * k3, rates)
    y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return torch.clamp_min(y_next, 0.0)


def rk4_step_substeps(y: torch.Tensor, dt: torch.Tensor, rates: torch.Tensor, n_sub: int) -> torch.Tensor:
    """
    RK4 with substepping for stability.

    Splits dt into n_sub equal substeps and applies RK4 n_sub times.
    """
    if n_sub <= 1:
        return rk4_step(y, dt, rates)

    h = dt / float(n_sub)
    y_cur = y
    for _ in range(n_sub):
        y_cur = rk4_step(y_cur, h, rates)
    return y_cur


# ---------------------------
# Model: hidden bolus as jump
# ---------------------------

class SimpleRNN(nn.Module):
    """
    Closed-loop kinetics learner:
      Inputs per step k: [u_k, y_prev] -> GRU -> params_k -> (jump) -> RK4

    Key change:
      - Instead of *basal production in the RHS*, we learn a *hidden jump*
        Δ_hidden_k ∈ R^P_{>=0} added at the same instant as the observed jump.

    Interpretation:
      The underlying 13D system receives boluses on species we don't observe.
      Their effect on the reduced observed state is approximated as an unknown
      instantaneous positive jump on the observed species.

    Returns:
      y_out     : (B,K,P) predicted y(t1..tK)
      rates_out : (B,K,8) predicted kinetic rates
      dhid_out  : (B,K,P) predicted hidden jump (diagnostics / regularization)
    """

    def __init__(
        self,
        in_u: int,
        P: int = 5,
        lift_dim: int = 32,
        hidden: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        u_to_y_jump: torch.Tensor | None = None,  # (U,P)
        n_sub: int = 10,
        # bounds
        rate_lo: float = 1e-2,
        rate_hi: float = 3.0,
        hiddenjump_lo: float = 0.0,
        hiddenjump_hi: float = 3.0,  # allow up to ~one typical bolus size per step
    ):
        super().__init__()
        self.P = int(P)
        self.in_u = int(in_u)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.n_sub = int(n_sub)

        self.rate_lo, self.rate_hi = float(rate_lo), float(rate_hi)
        self.hj_lo, self.hj_hi = float(hiddenjump_lo), float(hiddenjump_hi)

        in_dim = self.in_u + self.P  # [u_k, y_prev]

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

        # head -> 8 rates + P hidden jump
        self.head = nn.Linear(self.hidden, 8 + self.P)

        if u_to_y_jump is None:
            raise ValueError("u_to_y_jump must be provided.")
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
        nn.init.constant_(self.head.bias, 0.0)

    def forward(
        self,
        y0: torch.Tensor,     # (B,P)
        u_seq: torch.Tensor,  # (B,K,U)
        dt_seq: torch.Tensor, # (B,K)
        y_seq: torch.Tensor | None = None,  # (B,K,P) for teacher forcing
        teacher_forcing: bool = True,
        tf_every: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, K, U = u_seq.shape
        dev, dtype = u_seq.device, u_seq.dtype

        y_out = torch.empty(B, K, self.P, device=dev, dtype=dtype)
        rates_out = torch.empty(B, K, 8, device=dev, dtype=dtype)
        dhid_out = torch.empty(B, K, self.P, device=dev, dtype=dtype)

        y_prev = y0 + 0.01
        h = torch.zeros(self.num_layers, B, self.hidden, device=dev, dtype=dtype)

        for k in range(K):
            dt_k = dt_seq[:, k:k+1]   # (B,1)
            u_k = u_seq[:, k, :]      # (B,U)

            if teacher_forcing and (y_seq is not None) and k > 0 and (k % tf_every == 0):
                y_in = y_seq[:, k-1, :].detach()
            else:
                y_in = y_prev.detach()

            feat = torch.cat([u_k, y_in], dim=-1)
            z, h = self.gru(self.lift(feat).unsqueeze(1), h)
            z = z.squeeze(1)

            raw = self.head(z)  # (B, 8+P)

            # predicted kinetics
            rates = gamma(raw[:, 0:8], self.rate_lo, self.rate_hi)  # (B,8)

            # predicted hidden jump (>=0, bounded)
            dhid = gamma(raw[:, 8:8+self.P], self.hj_lo, self.hj_hi)  # (B,P)

            # Observed jump from measured control channels
            dobserved = u_k @ self.u_to_y_jump.to(device=dev, dtype=dtype)  # (B,P)

            # Total jump = observed + hidden (hidden approximates unobserved boluses)
            y_jump = y_prev + dobserved + dhid

            # Integrate
            y_next = rk4_step_substeps(y_jump, dt_k, rates, n_sub=self.n_sub)

            y_out[:, k, :] = y_next
            rates_out[:, k, :] = rates
            dhid_out[:, k, :] = dhid
            y_prev = y_next

        return (y_out, rates_out, dhid_out) if self.training else (y_out, rates_out.detach(), dhid_out.detach())


# ---------------------------
# Optional: simple regularizer
# ---------------------------

def hidden_jump_group_lasso(dhid_seq: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Group-lasso over time to encourage sparse-in-time hidden jumps.

    dhid_seq: (B,K,P)
    returns : scalar
    """
    # per-step L2 norm across species, then mean over B,K
    step_norm = torch.sqrt((dhid_seq ** 2).sum(dim=-1) + eps)  # (B,K)
    return step_norm.mean()
