from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def gamma(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    # bounded positive map in (lo, hi)
    return lo + (hi - lo) * torch.sigmoid(x)


def full_chain_rhs_torch(y: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Torch equivalent of benchmark_models.FullModel
    y: (B,13) [A, B, C, D, E, F, G, H, I, J, K, L, M]
    k: (B,19) [kf1-kf12, kr1, kr3, kr5, kr7, kr9, kr11, kr12]
    returns dy/dt: (B,13)
    """
    A, B, C, D, E, F, G, H, I, J, K, L, M = y.unbind(dim=-1)
    kf1, kf2, kf3, kf4, kf5, kf6, kf7, kf8, kf9, kf10, kf11, kf12, kr1, kr3, kr5, kr7, kr9, kr11, kr12 = k.unbind(dim=-1)

    dA = -kf1*A + kr1*B
    dB =  kf1*A - kr1*B - kf2*B
    dC =  kf2*B - kf3*C + kr3*D
    dD =  kf3*C - kr3*D - kf4*D
    dE =  kf4*D - kf5*E + kr5*F
    dF =  kf5*E - kr5*F - kf6*F
    dG =  kf6*F - kf7*G + kr7*H
    dH =  kf7*G - kr7*H - kf8*H
    dI =  kf8*H - kf9*I + kr9*J
    dJ =  kf9*I - kr9*J - kf10*J
    dK =  kf10*J - kf11*K + kr11*L
    dL =  kf11*K - kr11*L - kf12*L + kr12*M
    dM =  kf12*L - kr12*M

    return torch.stack([dA, dB, dC, dD, dE, dF, dG, dH, dI, dJ, dK, dL, dM], dim=-1)


def rk4_step(y: torch.Tensor, dt: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    One explicit RK4 step with frozen parameters k.
    y : (B,13)
    dt: (B,1)
    k : (B,19)
    """
    k1 = full_chain_rhs_torch(y, k)
    k2 = full_chain_rhs_torch(y + 0.5 * dt * k1, k)
    k3 = full_chain_rhs_torch(y + 0.5 * dt * k2, k)
    k4 = full_chain_rhs_torch(y + dt * k3, k)
    y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return torch.clamp_min(y_next, 0.0) # y can't be negative


def rk4_step_substeps(y: torch.Tensor, dt: torch.Tensor, k: torch.Tensor, n_sub: int = 50) -> torch.Tensor:
    """
    RK4 with multiple substeps for better numerical accuracy.
    Divides dt into n_sub smaller intervals and takes n_sub RK4 steps.
    
    y : (B,13)
    dt: (B,1)
    k : (B,19)
    n_sub: number of substeps (higher = more accurate, slower)
    """
    h = dt / float(n_sub)
    y_cur = y
    for _ in range(n_sub):
        y_cur = rk4_step(y_cur, h, k)
    return y_cur


class FullModelRNN(nn.Module):
    """
    Closed-loop kinetics learner for FULL 13-species model:
      - Observes P species (e.g., 5: A, D, G, J, M)
      - Internally simulates all 13 species
      - Learns 19 time-varying parameters
      
    Returns:
      out    : (B,K,P) predicted y(t1..tK) for P observed species
      params : (B,K,19) predicted theta_k (19 parameters for full model)
    """

    def __init__(
        self,
        in_u: int,
        P: int = 5, # number of observed species (e.g., 5 for A,D,G,J,M)
        obs_indices: list = None,  # which of the 13 species are observed (e.g., [0,3,6,9,12])
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
        
        # Map from observed indices to full 13-species state
        # Default: observe A, D, G, J, M (indices 0, 3, 6, 9, 12)
        if obs_indices is None:
            obs_indices = [0, 3, 6, 9, 12]
        self.obs_indices = obs_indices
        
        if len(obs_indices) != P:
            raise ValueError(f"obs_indices length {len(obs_indices)} must match P={P}")

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

        # head -> 19 parameters for full model
        self.head = nn.Linear(hidden, 19)

        # Jump map: converts bolus inputs u_k into an increment for the full state.
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
        theta_out = torch.empty(B, K, 19, device=dev, dtype=dtype)

        # Initialize full 13-species state (assume unobserved species start at small value)
        y_full_prev = torch.zeros(B, 13, device=dev, dtype=dtype)
        y_full_prev[:, self.obs_indices] = y0 + 0.01  # observed species
        y_full_prev[:, [i for i in range(13) if i not in self.obs_indices]] = 0.01  # unobserved

        h = torch.zeros(self.num_layers, B, self.hidden, device=dev, dtype=dtype)

        for k in range(K):
            dt_k = dt_seq[:, k:k+1]  # (B,1)
            u_k  = u_seq[:, k, :]    # (B,U)
            
            # Extract observed species from full state
            y_obs_prev = y_full_prev[:, self.obs_indices]

            # teacher forcing uses previous true observed state occasionally
            if teacher_forcing and (y_seq is not None) and k > 0 and (k % tf_every == 0):
                y_prev_in = y_seq[:, k-1, :].detach()
            else:
                y_prev_in = y_obs_prev.detach()

            # concatenate inputs directly
            feat = torch.cat([u_k, y_prev_in], dim=-1)
            x = self.lift(feat).unsqueeze(1)            # (B,1,H)

            # GRU update
            z, h = self.gru(x, h)                       # z: (B,1,H)
            z = z.squeeze(1)                            # (B,H)

            raw = self.head(z)                          # (B,19)

            # bounded rates - 12 forward, 7 reverse
            # Forward rates: kf1-kf12
            kf1  = gamma(raw[:, 0:1], 1e-3, 2.0)
            kf2  = gamma(raw[:, 1:2], 1e-3, 2.0)
            kf3  = gamma(raw[:, 2:3], 1e-3, 2.0)
            kf4  = gamma(raw[:, 3:4], 1e-3, 2.0)
            kf5  = gamma(raw[:, 4:5], 1e-3, 2.0)
            kf6  = gamma(raw[:, 5:6], 1e-3, 2.0)
            kf7  = gamma(raw[:, 6:7], 1e-3, 2.0)
            kf8  = gamma(raw[:, 7:8], 1e-3, 2.0)
            kf9  = gamma(raw[:, 8:9], 1e-3, 2.0)
            kf10 = gamma(raw[:, 9:10], 1e-3, 2.0)
            kf11 = gamma(raw[:, 10:11], 1e-3, 2.0)
            kf12 = gamma(raw[:, 11:12], 1e-3, 2.0)
            # Reverse rates: kr1, kr3, kr5, kr7, kr9, kr11, kr12
            kr1  = gamma(raw[:, 12:13], 1e-3, 2.0)
            kr3  = gamma(raw[:, 13:14], 1e-3, 2.0)
            kr5  = gamma(raw[:, 14:15], 1e-3, 2.0)
            kr7  = gamma(raw[:, 15:16], 1e-3, 2.0)
            kr9  = gamma(raw[:, 16:17], 1e-3, 2.0)
            kr11 = gamma(raw[:, 17:18], 1e-3, 2.0)
            kr12 = gamma(raw[:, 18:19], 1e-3, 2.0)
            
            theta_k = torch.cat([kf1, kf2, kf3, kf4, kf5, kf6, kf7, kf8, kf9, kf10, kf11, kf12,
                                kr1, kr3, kr5, kr7, kr9, kr11, kr12], dim=-1)  # (B,19)
            
            # Apply bolus jump to FULL 13-species state
            # u_to_y_jump maps control inputs to observed species only
            y_obs_jump = y_obs_prev + (u_k @ self.u_to_y_jump.to(device=dev, dtype=dtype))
            
            # Update full state with jumped observed species
            y_full_jump = y_full_prev.clone()
            y_full_jump[:, self.obs_indices] = y_obs_jump
            
            # Integrate full 13-species ODE
            y_full_next = rk4_step_substeps(y_full_jump, dt_k, theta_k, n_sub=1)
            
            # Extract observed species for output
            y_obs_next = y_full_next[:, self.obs_indices]
            
            y_out[:, k, :] = y_obs_next
            theta_out[:, k, :] = theta_k

            y_full_prev = y_full_next

        return (y_out, theta_out) if self.training else (y_out, theta_out.detach())
