from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn


def gamma(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


# =========================
# 7-state model: [A, D, G, J, K, L, M]
# =========================

def rhs_7_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    7-state reduced chain:
      states: [A, D, G, J, K, L, M]
      theta : (B,11) = [kfAD, kfDG, kfGJ, kf10, kf11, kf12,  krAD, krDG, krGJ, kr11, kr12]

    where:
      A <-> D   via kfAD, krAD   (lumps A->B->C->D)
      D <-> G   via kfDG, krDG   (lumps D->E->F->G)
      G <-> J   via kfGJ, krGJ   (lumps G->H->I->J)
      J -> K    via kf10
      K <-> L   via kf11, kr11
      L <-> M   via kf12, kr12
    """
    A, D, G, J, K, L, M = y.unbind(dim=-1)
    kfAD, kfDG, kfGJ, kf10, kf11, kf12, krAD, krDG, krGJ, kr11, kr12 = theta.unbind(dim=-1)

    dA = -kfAD * A + krAD * D
    dD =  kfAD * A - krAD * D - kfDG * D + krDG * G
    dG =  kfDG * D - krDG * G - kfGJ * G + krGJ * J
    dJ =  kfGJ * G - krGJ * J - kf10 * J
    dK =  kf10 * J - kf11 * K + kr11 * L
    dL =  kf11 * K - kr11 * L - kf12 * L + kr12 * M
    dM =  kf12 * L - kr12 * M

    return torch.stack([dA, dD, dG, dJ, dK, dL, dM], dim=-1)


def rk4_step_7(y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    k1 = rhs_7_torch(y, theta)
    k2 = rhs_7_torch(y + 0.5 * dt * k1, theta)
    k3 = rhs_7_torch(y + 0.5 * dt * k2, theta)
    k4 = rhs_7_torch(y + dt * k3, theta)
    y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return torch.clamp_min(y_next, 0.0)


def rk4_step_substeps_7(y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor, n_sub: int) -> torch.Tensor:
    if n_sub <= 1:
        return rk4_step_7(y, dt, theta)
    h = dt / float(n_sub)
    y_cur = y
    for _ in range(n_sub):
        y_cur = rk4_step_7(y_cur, h, theta)
    return y_cur


class SimpleRNN7(nn.Module):
    """
    Closed-loop kinetics learner for 7-state model:
      inputs: [u_k, y_obs_prev] -> GRU -> theta_k -> jump in FULL state -> RK4

    We keep the interface consistent with your previous script:
      y0_obs : (B, n_obs)
      u_seq  : (B, K, U)
      dt_seq : (B, K)
      y_seq  : (B, K, n_obs) optional teacher forcing

    Outputs:
      y_hat_obs : (B, K, n_obs)
      theta_hat : (B, K, 11)

    Internally we simulate a latent 7D state. You must provide:
      - obs_indices: which coordinates of the 7D state correspond to observed channels (length n_obs)
      - u_to_x_jump: map from control bolus u to FULL latent state jump (U x 7)
      - x0_from_y0: either provide full x0 (B,7) directly, or we embed observed y0 into x0 with zeros elsewhere.
    """

    def __init__(
        self,
        in_u: int,
        obs_indices: torch.Tensor,          # (n_obs,)
        P_full: int = 7,
        lift_dim: int = 32,
        hidden: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        u_to_x_jump: torch.Tensor | None = None,  # (U,7)
        n_sub: int = 10,
        rate_lo: float = 1e-2,
        rate_hi: float = 3.0,
    ):
        super().__init__()
        self.P_full = int(P_full)
        self.in_u = int(in_u)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.n_sub = int(n_sub)
        self.rate_lo, self.rate_hi = float(rate_lo), float(rate_hi)

        obs_indices = obs_indices.to(dtype=torch.long)
        self.register_buffer("obs_indices", obs_indices, persistent=True)
        self.n_obs = int(obs_indices.numel())

        if u_to_x_jump is None:
            raise ValueError("u_to_x_jump must be provided (shape U x 7).")
        self.register_buffer("u_to_x_jump", u_to_x_jump.to(dtype=torch.float32), persistent=True)

        in_dim = self.in_u + self.n_obs

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

        self.head = nn.Linear(self.hidden, 11)

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

    def _embed_y0(self, y0_obs: torch.Tensor) -> torch.Tensor:
        B = y0_obs.shape[0]
        x0 = torch.zeros(B, self.P_full, device=y0_obs.device, dtype=y0_obs.dtype)
        x0[:, self.obs_indices] = y0_obs
        return x0

    def forward(
        self,
        y0_obs: torch.Tensor,     # (B, n_obs)
        u_seq: torch.Tensor,      # (B, K, U)
        dt_seq: torch.Tensor,     # (B, K)
        y_seq: torch.Tensor | None = None,  # (B, K, n_obs)
        teacher_forcing: bool = True,
        tf_every: int = 50,
        x0_full: torch.Tensor | None = None,  # optionally pass (B,7) full init
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, K, U = u_seq.shape
        dev, dtype = u_seq.device, u_seq.dtype

        y_hat = torch.empty(B, K, self.n_obs, device=dev, dtype=dtype)
        theta_hat = torch.empty(B, K, 11, device=dev, dtype=dtype)

        x_prev = self._embed_y0(y0_obs) if x0_full is None else x0_full.to(device=dev, dtype=dtype)
        x_prev = x_prev + 0.01

        h = torch.zeros(self.num_layers, B, self.hidden, device=dev, dtype=dtype)

        for k in range(K):
            dt_k = dt_seq[:, k:k+1]
            u_k = u_seq[:, k, :]

            x_obs_prev = x_prev[:, self.obs_indices]

            if teacher_forcing and (y_seq is not None) and k > 0 and (k % tf_every == 0):
                x_obs_in = y_seq[:, k-1, :].detach()
            else:
                x_obs_in = x_obs_prev.detach()

            feat = torch.cat([u_k, x_obs_in], dim=-1)
            z = self.lift(feat).unsqueeze(1)
            z, h = self.gru(z, h)
            z = z.squeeze(1)

            raw = self.head(z)
            theta_k = gamma(raw, self.rate_lo, self.rate_hi)  # (B,11)

            x_jump = x_prev + (u_k @ self.u_to_x_jump.to(device=dev, dtype=dtype))
            x_next = rk4_step_substeps_7(x_jump, dt_k, theta_k, n_sub=self.n_sub)

            y_hat[:, k, :] = x_next[:, self.obs_indices]
            theta_hat[:, k, :] = theta_k
            x_prev = x_next

        return (y_hat, theta_hat) if self.training else (y_hat, theta_hat.detach())


# =========================
# 9-state model: [A, D, G, H, I, J, K, L, M]
# =========================

def rhs_9_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    9-state reduced chain:
      states: [A, D, G, H, I, J, K, L, M]
      theta : (B,14) = [kfAD, kfDG, kf7, kf8, kf9, kf10, kf11, kf12,
                       krAD, krDG, kr7, kr9, kr11, kr12]

    where:
      A <-> D   lumped
      D <-> G   lumped
      G <-> H   via kf7, kr7
      H -> I    via kf8
      I <-> J   via kf9, kr9
      J -> K    via kf10
      K <-> L   via kf11, kr11
      L <-> M   via kf12, kr12
    """
    A, D, G, H, I, J, K, L, M = y.unbind(dim=-1)

    (kfAD, kfDG, kf7, kf8, kf9, kf10, kf11, kf12,
     krAD, krDG, kr7, kr9, kr11, kr12) = theta.unbind(dim=-1)

    dA = -kfAD * A + krAD * D
    dD =  kfAD * A - krAD * D - kfDG * D + krDG * G

    dG =  kfDG * D - krDG * G - kf7 * G + kr7 * H
    dH =  kf7 * G - kr7 * H - kf8 * H
    dI =  kf8 * H - kf9 * I + kr9 * J
    dJ =  kf9 * I - kr9 * J - kf10 * J

    dK =  kf10 * J - kf11 * K + kr11 * L
    dL =  kf11 * K - kr11 * L - kf12 * L + kr12 * M
    dM =  kf12 * L - kr12 * M

    return torch.stack([dA, dD, dG, dH, dI, dJ, dK, dL, dM], dim=-1)


def rk4_step_9(y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    k1 = rhs_9_torch(y, theta)
    k2 = rhs_9_torch(y + 0.5 * dt * k1, theta)
    k3 = rhs_9_torch(y + 0.5 * dt * k2, theta)
    k4 = rhs_9_torch(y + dt * k3, theta)
    y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return torch.clamp_min(y_next, 0.0)


def rk4_step_substeps_9(y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor, n_sub: int) -> torch.Tensor:
    if n_sub <= 1:
        return rk4_step_9(y, dt, theta)
    h = dt / float(n_sub)
    y_cur = y
    for _ in range(n_sub):
        y_cur = rk4_step_9(y_cur, h, theta)
    return y_cur


class SimpleRNN9(nn.Module):
    """
    Same as SimpleRNN7 but with a 9D latent state and 14 rates.
    """

    def __init__(
        self,
        in_u: int,
        obs_indices: torch.Tensor,          # (n_obs,)
        P_full: int = 9,
        lift_dim: int = 32,
        hidden: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        u_to_x_jump: torch.Tensor | None = None,  # (U,9)
        n_sub: int = 10,
        rate_lo: float = 1e-2,
        rate_hi: float = 3.0,
    ):
        super().__init__()
        self.P_full = int(P_full)
        self.in_u = int(in_u)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.n_sub = int(n_sub)
        self.rate_lo, self.rate_hi = float(rate_lo), float(rate_hi)

        obs_indices = obs_indices.to(dtype=torch.long)
        self.register_buffer("obs_indices", obs_indices, persistent=True)
        self.n_obs = int(obs_indices.numel())

        if u_to_x_jump is None:
            raise ValueError("u_to_x_jump must be provided (shape U x 9).")
        self.register_buffer("u_to_x_jump", u_to_x_jump.to(dtype=torch.float32), persistent=True)

        in_dim = self.in_u + self.n_obs

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

        self.head = nn.Linear(self.hidden, 14)

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

    def _embed_y0(self, y0_obs: torch.Tensor) -> torch.Tensor:
        B = y0_obs.shape[0]
        x0 = torch.zeros(B, self.P_full, device=y0_obs.device, dtype=y0_obs.dtype)
        x0[:, self.obs_indices] = y0_obs
        return x0

    def forward(
        self,
        y0_obs: torch.Tensor,     # (B, n_obs)
        u_seq: torch.Tensor,      # (B, K, U)
        dt_seq: torch.Tensor,     # (B, K)
        y_seq: torch.Tensor | None = None,  # (B, K, n_obs)
        teacher_forcing: bool = True,
        tf_every: int = 50,
        x0_full: torch.Tensor | None = None,  # optionally pass (B,9) full init
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, K, U = u_seq.shape
        dev, dtype = u_seq.device, u_seq.dtype

        y_hat = torch.empty(B, K, self.n_obs, device=dev, dtype=dtype)
        theta_hat = torch.empty(B, K, 14, device=dev, dtype=dtype)

        x_prev = self._embed_y0(y0_obs) if x0_full is None else x0_full.to(device=dev, dtype=dtype)
        x_prev = x_prev + 0.01

        h = torch.zeros(self.num_layers, B, self.hidden, device=dev, dtype=dtype)

        for k in range(K):
            dt_k = dt_seq[:, k:k+1]
            u_k = u_seq[:, k, :]

            x_obs_prev = x_prev[:, self.obs_indices]

            if teacher_forcing and (y_seq is not None) and k > 0 and (k % tf_every == 0):
                x_obs_in = y_seq[:, k-1, :].detach()
            else:
                x_obs_in = x_obs_prev.detach()

            feat = torch.cat([u_k, x_obs_in], dim=-1)
            z = self.lift(feat).unsqueeze(1)
            z, h = self.gru(z, h)
            z = z.squeeze(1)

            raw = self.head(z)
            theta_k = gamma(raw, self.rate_lo, self.rate_hi)  # (B,14)

            x_jump = x_prev + (u_k @ self.u_to_x_jump.to(device=dev, dtype=dtype))
            x_next = rk4_step_substeps_9(x_jump, dt_k, theta_k, n_sub=self.n_sub)

            y_hat[:, k, :] = x_next[:, self.obs_indices]
            theta_hat[:, k, :] = theta_k
            x_prev = x_next

        return (y_hat, theta_hat) if self.training else (y_hat, theta_hat.detach())