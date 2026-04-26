from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaffolds import MechanisticScaffold


def gamma(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


# def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
#     # Linear alternative (DO NOT USE for wide bounds):
#     #   return lo + (hi - lo) * torch.sigmoid(x)
#     # At init (x≈0, sigmoid≈0.5) this gives arithmetic midpoint (lo+hi)/2.
#     # For bounds like knuc_A=[0.1,100] that's 50 — 5× true value, causing ODE blowup.
#     # Log-sigmoid gives geometric midpoint sqrt(lo*hi) ≈ 3.2 for knuc_A, which is stable.
#     return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x))

def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    # Sigmoid: strictly bounded in [lo, hi], initialises at geometric mean sqrt(lo*hi) when x=0.
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x))


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
        gru_u_cols: Optional[list] = None,  # which u columns enter the GRU (None = all)
        head_bias_init: float = 0.0,        # init all head biases to this value (<0 starts theta near lo)
        head_weight_gain: float = 1.0,      # Xavier gain for head weights (>1 amplifies per-experiment variation)
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
        self.gru_u_cols   = list(gru_u_cols) if gru_u_cols is not None else None

        # Per-parameter bounds — use scaffold-defined if available, else broadcast scalar
        if rhs.theta_lo_vec is not None and rhs.theta_hi_vec is not None:
            lo = torch.tensor(rhs.theta_lo_vec, dtype=torch.float32)
            hi = torch.tensor(rhs.theta_hi_vec, dtype=torch.float32)
        else:
            lo = torch.full((self.theta_dim,), theta_lo)
            hi = torch.full((self.theta_dim,), theta_hi)
        self.register_buffer("theta_lo_vec", lo)
        self.register_buffer("theta_hi_vec", hi)

        gru_feat_dim = len(self.gru_u_cols) if self.gru_u_cols is not None else self.U
        self.lift = nn.Sequential(
            nn.Linear(gru_feat_dim + self.P, lift_dim),
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
        if head_bias_init != 0.0:
            nn.init.constant_(self.head.bias, float(head_bias_init))
        if head_weight_gain != 1.0:
            nn.init.xavier_uniform_(self.head.weight, gain=float(head_weight_gain))

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
        tbptt_chunk: int = 0,                 # if >0, detach h every tbptt_chunk steps (truncated BPTT)
        u_transform: str = "none",            # GRU input transform: "none" | "cumsum" | "sqrt" | "cumsum_sqrt"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        y_out    = torch.empty(B, K, self.P, device=y0.device, dtype=y0.dtype)
        th_out   = torch.empty(B, K, self.theta_dim, device=y0.device, dtype=y0.dtype)
        beta_out = torch.zeros(B, K, self.P, device=y0.device, dtype=y0.dtype)

        h = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size, device=y0.device, dtype=y0.dtype)

        use_partial = obs_idx.numel() > 0

        # Pre-compute the GRU's view of u_seq (separate from the raw delta used for ODE jumps).
        # cumsum: after a bolus at step t, the GRU sees it at ALL subsequent steps — no long-range
        # memory required. The ODE jump always uses the raw delta u_seq.
        if u_transform in ("cumsum", "cumsum_sqrt"):
            u_gru = u_seq.cumsum(dim=1)
        else:
            u_gru = u_seq
        if u_transform in ("sqrt", "cumsum_sqrt"):
            u_gru = u_gru.clamp_min(0.0).sqrt()

        y_prev = y0
        for k in range(K):
            u_k     = u_seq[:, k, :]   # raw delta — used only for ODE jumps
            u_gru_k = u_gru[:, k, :]   # transformed — used for GRU features
            dt_k = dt_seq[:, k]

            # Truncated BPTT: detach hidden state every tbptt_chunk steps so that
            # gradients stay local and don't vanish over the full trajectory length.
            if tbptt_chunk > 0 and k > 0 and (k % tbptt_chunk == 0):
                h = h.detach()

            y_in = y_prev.detach()

            if teacher_forcing and k > 0 and (k % tf_every == 0) and y_seq is not None:
                if use_partial:
                    y_in = y_prev.clone()
                    idx = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].to(dtype=y_in.dtype).detach()
                else:
                    y_in = y_seq[:, k - 1, :].to(dtype=y_prev.dtype).detach()

            u_gru_k_feat = u_gru_k[:, self.gru_u_cols] if self.gru_u_cols is not None else u_gru_k
            feat = torch.cat([u_gru_k_feat, y_in], dim=-1)
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
                y = self._rk4_substeps_basal(y, dt_k, theta_k, beta_k)
            else:
                if self.theta_bounded:
                    theta_k = log_gamma(raw, self.theta_lo_vec, self.theta_hi_vec)
                else:
                    theta_k = F.softplus(raw)
                y = y_prev + (u_k @ self.u_to_y_jump)
                y = self._rk4_substeps(y, dt_k, theta_k)

            y_out[:, k, :] = y
            th_out[:, k, :] = theta_k
            y_prev = y

        return y_out, th_out, beta_out

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
            k4 = rhs(y +       hdt * k3,  theta)
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
            k4 = rhs(y +       hdt * k3,  theta) + beta
            y  = y + (hdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.clamp_min(y, 0.0)
