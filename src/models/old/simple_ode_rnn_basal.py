from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn


def gamma(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    # bounded positive map in (lo, hi)
    return lo + (hi - lo) * torch.sigmoid(x)


def reduced_chain_rhs_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Reduced 5-state chain with *basal production* terms to soak up unmodelled mass
    (e.g. boluses into unobserved species in the underlying 13D system).

    y    : (B,5)  [A, D, G, J, M]
    theta: (B,13) [kf1,kf2,kf3,kf4, kr1,kr2,kr3,kr4,  kj1,kj2,kj3,kj4,kj5]
                  where kj* are nonnegative "basal production / source" terms.

    returns dy/dt: (B,5)
    """
    A, D, G, J, M = y.unbind(dim=-1)

    kf1, kf2, kf3, kf4, kr1, kr2, kr3, kr4, kj1, kj2, kj3, kj4, kj5 = theta.unbind(dim=-1)

    # Chain with extra source terms
    dA = -kf1 * A + kr1 * D + kj1
    dD =  kf1 * A - kr1 * D - kf2 * D + kr2 * G + kj2
    dG =  kf2 * D - kr2 * G - kf3 * G + kr3 * J + kj3
    dJ =  kf3 * G - kr3 * J - kf4 * J + kr4 * M + kj4
    dM =  kf4 * J - kr4 * M + kj5

    return torch.stack([dA, dD, dG, dJ, dM], dim=-1)


def rk4_step(y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    One explicit RK4 step with frozen parameters theta.

    y    : (B,5)
    dt   : (B,1)
    theta: (B,13)
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

    Key change vs earlier version:
      - theta_k now includes *basal production* terms kj1..kj5 to absorb the
        effect of unobserved boluses / missing mass in the reduced model.

    Returns:
      out    : (B,K,P) predicted y(t1..tK)
      params : (B,K,13) predicted theta_k = [rates(8), basal(5)]
    """

    def __init__(
        self,
        in_u: int,
        P: int = 5,  # number of observed species
        lift_dim: int = 32,
        hidden: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        u_to_y_jump: torch.Tensor | None = None,  # shape (U,P)
        n_sub: int = 10,  # RK4 substeps — must be >= 10 for stability (dt≈0.5, rates up to 3.0)
        # bounds for parameters
        rate_lo: float = 1e-2,
        rate_hi: float = 3.0,
        basal_lo: float = 0.0,
        basal_hi: float = 0.3,  # ~10% of a typical bolus per time unit; keeps steady-state drift bounded
    ):
        super().__init__()
        self.P = P
        self.in_u = in_u
        self.hidden = hidden
        self.num_layers = num_layers
        self.n_sub = int(n_sub)

        self.rate_lo, self.rate_hi = float(rate_lo), float(rate_hi)
        self.basal_lo, self.basal_hi = float(basal_lo), float(basal_hi)

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

        # head -> 13 params: 8 rates + 5 basal productions
        self.head = nn.Linear(hidden, 13)

        # Jump map: converts bolus inputs u_k into an increment for the reduced state.
        # Note: only observed species can be jumped, by design.
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
        theta_out = torch.empty(B, K, 13, device=dev, dtype=dtype)

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

            # concatenate inputs directly
            feat = torch.cat([u_k, y_prev_in], dim=-1)
            x = self.lift(feat).unsqueeze(1)     # (B,1,H)
            z, h = self.gru(x, h)                # z: (B,1,H)
            z = z.squeeze(1)                     # (B,H)

            raw = self.head(z)                   # (B,13)

            # 8 kinetic rates (positive, bounded)
            rates = gamma(raw[:, 0:8], self.rate_lo, self.rate_hi)

            # 5 basal/source terms (nonnegative, bounded)
            basal = gamma(raw[:, 8:13], self.basal_lo, self.basal_hi)

            theta_k = torch.cat([rates, basal], dim=-1)  # (B,13)

            # quick diagnostics (won't TorchScript, but fine for eager runs)
            # if not torch.isfinite(theta_k).all():
            #     print(f"[k={k}] theta nan/inf  min={theta_k.min().item():.3g}  max={theta_k.max().item():.3g}")

            # Apply bolus jump (only on observed species)
            y_jump = y_prev + (u_k @ self.u_to_y_jump.to(device=dev, dtype=dtype))
            # if not torch.isfinite(y_jump).all():
            #     print(f"[k={k}] y_jump nan/inf  min={y_jump.min().item():.3g}  max={y_jump.max().item():.3g}")

            # Integrate with RK4 substepping
            y_next = rk4_step_substeps(y_jump, dt_k, theta_k, n_sub=self.n_sub)
            # if not torch.isfinite(y_next).all():
            #     print(f"[k={k}] y_next nan/inf after RK4  min={y_next.min().item():.3g}  max={y_next.max().item():.3g}")

            y_out[:, k, :] = y_next
            theta_out[:, k, :] = theta_k
            y_prev = y_next

        return (y_out, theta_out) if self.training else (y_out, theta_out.detach())
