from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaffolds import MechanisticScaffold


def gamma(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


# --- OLD log_gamma (Sigmoid-based, prone to vanishing gradients at bounds) ---
# def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
#     return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x))


# --- NEW log_gamma (Softplus-based, from supervisor's log_uniform hint) ---
def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    # Softplus provides a gradient of 1 for large values, eliminating the bottleneck.
    # Note: 'hi' now acts as a scaling reference rather than a strict hard clamp, 
    # preventing the parameter from getting irreversibly stuck at the exact boundary.
    z = F.softplus(x)
    return lo * torch.exp(torch.log(hi / lo + eps) * z)


class OdeRNN(nn.Module):
    """
    Closed-loop mechanistic model:
      (u_k, y_{k-1}) -> GRU -> theta_k
      y <- y + u_k @ jump
      y <- integrate ODE(scaffold, theta_k) over dt_k
    """

    def __init__(
        self,
        *,
        U: int,
        rhs: MechanisticScaffold,
        u_to_y_jump: torch.Tensor,   # (U,P)
        hidden: int = 128,
        lift_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        theta_lo: float = 1e-3,
        theta_hi: float = 2.0,
        n_substeps: int = 1,
        use_basal: bool = False,
        theta_bounded: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.U = int(U)
        self.P = int(rhs.P)
        self.theta_dim = int(rhs.theta_dim)
        self.rhs = rhs
        self.n_substeps   = int(n_substeps)
        self.use_basal    = bool(use_basal)
        self.theta_lo     = float(theta_lo)
        self.theta_hi     = float(theta_hi)
        self.theta_bounded = bool(theta_bounded)

        # Per-parameter bounds — use scaffold-defined if available, else broadcast scalar
        if rhs.theta_lo_vec is not None and rhs.theta_hi_vec is not None:
            lo = torch.tensor(rhs.theta_lo_vec, dtype=torch.float32)
            hi = torch.tensor(rhs.theta_hi_vec, dtype=torch.float32)
        else:
            lo = torch.full((self.theta_dim,), theta_lo)
            hi = torch.full((self.theta_dim,), theta_hi)
        self.register_buffer("theta_lo_vec", lo)
        self.register_buffer("theta_hi_vec", hi)

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
        head_out = self.theta_dim + self.P if self.use_basal else self.theta_dim
        self.head = nn.Linear(hidden, head_out)

        if u_to_y_jump.shape != (self.U, self.P):
            raise ValueError(f"u_to_y_jump must be (U,P)=({self.U},{self.P}), got {tuple(u_to_y_jump.shape)}")
        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

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
        y_out    = torch.empty(B, K, self.P, device=y0.device, dtype=y0.dtype)
        th_out   = torch.empty(B, K, self.theta_dim, device=y0.device, dtype=y0.dtype)
        beta_out = torch.zeros(B, K, self.P, device=y0.device, dtype=y0.dtype)

        h = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size, device=y0.device, dtype=y0.dtype)

        use_partial = obs_idx.numel() > 0

        y_prev = y0
        for k in range(K):
            u_k  = u_seq[:, k, :]
            dt_k = dt_seq[:, k]

            y_in = y_prev.detach()

            if teacher_forcing and k > 0 and (k % tf_every == 0) and y_seq is not None:
                if use_partial:
                    y_in = y_prev.clone()
                    idx = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].to(dtype=y_in.dtype).detach()
                else:
                    y_in = y_seq[:, k - 1, :].to(dtype=y_prev.dtype).detach()

            feat = torch.cat([u_k, y_in], dim=-1)
            x = self.lift(feat).unsqueeze(1)
            z, h = self.gru(x, h)
            raw = self.head(z.squeeze(1))

            if self.use_basal:
                raw_theta = raw[:, :self.theta_dim]
                if self.theta_bounded:
                    theta_k = log_gamma(raw_theta, self.theta_lo_vec, self.theta_hi_vec)
                else:
                    theta_k = F.softplus(raw_theta)
                beta_k = raw[:, self.theta_dim:] * (y_prev / (y_prev + 1.0))
                beta_out[:, k, :] = beta_k
                y = y_prev + (u_k @ self.u_to_y_jump)
                
                # --- INTEGRATOR CALL ---
                y = self._rk4_substeps_basal(y, dt_k, theta_k, beta_k)
                # Note: If you adapt analytical step for basal, swap here too.
            else:
                if self.theta_bounded:
                    theta_k = log_gamma(raw, self.theta_lo_vec, self.theta_hi_vec)
                else:
                    theta_k = F.softplus(raw)
                y = y_prev + (u_k @ self.u_to_y_jump)
                
                # =======================================================
                # INTEGRATOR TOGGLE
                # =======================================================
                # 1. OLD RK4 (Commented out)
                # y = self._rk4_substeps(y, dt_k, theta_k)
                
                # 2. NEW ANALYTICAL STEP
                y = self._analytical_step(y, dt_k, theta_k)
                # =======================================================

            y_out[:, k, :] = y
            th_out[:, k, :] = theta_k
            y_prev = y

        return y_out, th_out, beta_out

    # --- NEW ANALYTICAL INTEGRATOR ---
    def _analytical_step(
        self,
        y: torch.Tensor,
        dt: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Analytical step based on the supervisor's exact integration.
        Assumes 6-state scaffold layout: [R, m, mm, p, pm, DNA]
        """
        eps = 1e-8
        # 1. Unbind the states
        R_prev, m_prev, mm_prev, p_prev, pm_prev, DNA_curr = y.unbind(dim=-1)
        
        # 2. Unbind your parameters (YOU MUST MATCH THESE TO YOUR SCAFFOLD'S THETA DIMENSIONS)
        # Example format based on typical TXTL scaffolds:
        # VTX, KTX, VTL, kdm, kmt, ... = theta.unbind(dim=-1)
        
        # ---------------------------------------------------------
        # TODO: Assign correct indices from theta here:
        kmt = theta[..., 0]   # Example index
        kdm = theta[..., 1]   # Example index
        VTL_eff = theta[..., 2] # Example index (maybe VTL * R_prev / (R_prev + K))
        M_inf = theta[..., 3] # Example index (Steady state RNA)
        # ---------------------------------------------------------
        
        # 3. Supervisor's Analytical Integration Logic
        eta    = torch.exp(-kmt * dt)       
        delta  = (kdm - kmt)
        same   = torch.abs(delta) < 1e-6

        exp_M = torch.exp(-kdm * dt)

        int_M_eq  = (M_inf * (1.0 - eta) / (kmt + eps)
                     + (m_prev - M_inf) * dt * eta)
        int_M_gen = (M_inf * (1.0 - eta) / (kmt + eps)
                     + (m_prev - M_inf) * (eta - exp_M) / (delta + eps))
        int_M_conv = torch.where(same, int_M_eq, int_M_gen)

        # Calculate new states (Add your specific RNA/Broccoli logic here)
        m_curr = m_prev # Replace with analytical RNA equation
        mm_curr = mm_prev # Replace with analytical Broccoli equation
        R_curr = R_prev # Replace with analytical R depletion equation

        p_curr  = torch.clamp_min(p_prev * eta + VTL_eff * int_M_conv, 0.0)

        int_M_total = (M_inf * dt
                      + (m_prev - M_inf) / (kdm + eps) * (1.0 - exp_M))
        
        pm_curr = torch.clamp_min(pm_prev + (p_prev - p_curr) + VTL_eff * int_M_total, 0.0)

        return torch.stack([R_curr, m_curr, mm_curr, p_curr, pm_curr, DNA_curr], dim=-1)

    # --- OLD RK4 INTEGRATOR ---
    # def _rk4_substeps(
    #     self,
    #     y: torch.Tensor,
    #     dt: torch.Tensor,
    #     theta: torch.Tensor,
    # ) -> torch.Tensor:
    #     rhs = self.rhs
    #     n_sub = self.n_substeps
    #     dt = dt.unsqueeze(1)
    #     hdt = dt / float(n_sub)
    #     for _ in range(n_sub):
    #         k1 = rhs(y,                   theta)
    #         k