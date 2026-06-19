"""
OdeMambapySSM: Mamba encoder using mambapy (pure-PyTorch, CPU/MPS/CUDA compatible).

Drop-in replacement for OdeRNN. No CUDA kernels required.
Install: pip install mambapy

Uses MambaBlock.step() for a fully differentiable recurrent step:
  - cache = (h, inputs): h is (B, ED, N), inputs is (B, ED, d_conv-1)
  - h starts as None (zero-initialised on first step)
  - gradients flow through h across steps

Mirrors the OdeLSTM/OdeRNN API: supports gru_u_cols/y_cols subsetting,
u_transform/y_transform, analytic-scaffold path (e.g. IvttAnalyticScaffold),
and the same teacher-forcing schedule.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mambapy.mamba import MambaBlock, MambaConfig

from scaffolds import MechanisticScaffold
from models.u_features import u_feature_mult, build_u_enc


def gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x / tau))


class OdeMambapySSM(nn.Module):
    """
    Mamba SSM encoder for mechanistic ODE parameter inference.

    Architecture per step:
      (u_k, y_{k-1}) -> lift -> MambaBlocks (recurrent step) -> head -> theta_k
      y <- y + u_k @ jump            (or analytic_step for analytic scaffolds)
      y <- RK4(scaffold, theta_k, dt_k)
    """

    def __init__(
        self,
        *,
        U: int,
        rhs: MechanisticScaffold,
        u_to_y_jump: torch.Tensor,
        hidden: int = 128,
        lift_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.0,
        theta_lo: float = 1e-3,
        theta_hi: float = 2.0,
        n_substeps: int = 1,
        use_basal: bool = False,
        theta_bounded: bool = True,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        gru_u_cols: Optional[list] = None,
        gru_y_cols: Optional[list] = None,
        lift_skip: bool = False,
        head_init: str = "default",
        theta_head_transform: str = "log_gamma",
        theta_head_tau: float = 1.0,
        u_transform: str = "none",   # encoder u-feature transform (sizes the lift)
        **kwargs,
    ):
        super().__init__()
        self.U = int(U)
        self.P = int(rhs.P)
        self.theta_dim = int(rhs.theta_dim)
        self.rhs = rhs
        self.n_substeps = int(n_substeps)
        self.use_basal = bool(use_basal)
        self.theta_bounded = bool(theta_bounded)
        self.hidden = int(hidden)
        n = max(1, num_layers)

        if theta_head_transform not in ("log_gamma", "gamma"):
            raise ValueError(f"theta_head_transform must be 'log_gamma' or 'gamma', got {theta_head_transform}")
        self.theta_head_transform = str(theta_head_transform)
        self.theta_head_tau = float(theta_head_tau)

        self._analytic_scaffold = bool(getattr(rhs, "has_analytic_step", False))
        self._tf_at_k_zero = bool(getattr(rhs, "tf_at_k_zero", False))
        self.theta_dim_emit = int(getattr(rhs, "theta_dim_emit", self.theta_dim))

        self.gru_u_cols = list(gru_u_cols) if gru_u_cols is not None else None
        self.gru_y_cols = list(gru_y_cols) if gru_y_cols is not None else None
        u_cols_dim = len(self.gru_u_cols) if self.gru_u_cols is not None else self.U
        y_cols_dim = len(self.gru_y_cols) if self.gru_y_cols is not None else self.P

        self._u_transform = str(u_transform)
        self._u_mult = u_feature_mult(self._u_transform)
        self._has_u_cols = self.gru_u_cols is not None
        self.register_buffer(
            "gru_u_idx",
            torch.tensor(self.gru_u_cols if self.gru_u_cols is not None else [], dtype=torch.long),
            persistent=False,
        )

        if rhs.theta_lo_vec is not None and rhs.theta_hi_vec is not None:
            lo = torch.tensor(rhs.theta_lo_vec, dtype=torch.float32)
            hi = torch.tensor(rhs.theta_hi_vec, dtype=torch.float32)
        else:
            lo = torch.full((self.theta_dim,), theta_lo)
            hi = torch.full((self.theta_dim,), theta_hi)
        self.register_buffer("theta_lo_vec", lo)
        self.register_buffer("theta_hi_vec", hi)

        self.lift_skip = bool(lift_skip)
        feat_in = u_cols_dim * self._u_mult + y_cols_dim
        # Mamba expects a fixed d_model — if lift_skip, use a single Linear to project
        # feat_in into the model dim; otherwise the standard MLP lift.
        if self.lift_skip:
            self.lift = nn.Linear(feat_in, hidden)
        else:
            self.lift = nn.Sequential(
                nn.Linear(feat_in, lift_dim),
                nn.SiLU(),
                nn.Linear(lift_dim, hidden),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )

        cfg = MambaConfig(
            d_model=hidden,
            n_layers=1,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand,
            pscan=False,
            use_cuda=False,
        )
        self.mamba_layers = nn.ModuleList([MambaBlock(cfg) for _ in range(n)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n)])

        head_out = self.theta_dim + self.P if use_basal else self.theta_dim
        self.head = nn.Linear(hidden, head_out)
        if head_init not in ("default", "orthogonal", "supervisor"):
            raise ValueError(f"head_init must be 'default' or 'orthogonal', got {head_init}")
        # "supervisor" is a backward-compatible alias for "orthogonal".
        if str(head_init) in ("orthogonal", "supervisor"):
            nn.init.xavier_uniform_(self.head.weight, gain=1.0)
            nn.init.zeros_(self.head.bias)
        else:
            nn.init.normal_(self.head.weight, std=0.01)
            nn.init.zeros_(self.head.bias)

        if u_to_y_jump.shape != (self.U, self.P):
            raise ValueError(
                f"u_to_y_jump must be (U,P)=({self.U},{self.P}), got {tuple(u_to_y_jump.shape)}"
            )
        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

    def forward(
        self,
        y0: torch.Tensor,
        u_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        obs_idx: torch.Tensor,
        y_seq: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True,
        tf_every: int = 50,
        u_transform: str = "none",
        y_transform: str = "none",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        device, dtype = y0.device, y0.dtype

        y_out = torch.empty(B, K, self.P, device=device, dtype=dtype)
        th_out = torch.empty(B, K, self.theta_dim_emit, device=device, dtype=dtype)
        beta_out = torch.zeros(B, K, self.P, device=device, dtype=dtype)

        caches = self._init_caches(B, device, dtype)
        use_partial = obs_idx.numel() > 0

        analytic_ctx: dict = {}
        if self._analytic_scaffold:
            analytic_ctx = self.rhs.precompute_batch(y0, u_seq)
            y_prev = self.rhs.initial_state(y0)
        else:
            y_prev = y0

        u_enc = build_u_enc(u_seq, dt_seq, self._u_transform, self.gru_u_idx, self._has_u_cols)

        for k in range(K):
            u_k = u_seq[:, k, :]
            u_feat = u_enc[:, k, :]   # cols + transform already applied
            dt_k = dt_seq[:, k]

            y_in = y_prev.detach()
            tf_fires = (k % tf_every == 0) if self._tf_at_k_zero else (k > 0 and k % tf_every == 0)
            if teacher_forcing and tf_fires and y_seq is not None:
                if use_partial:
                    y_in = y_prev.clone()
                    idx = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].to(dtype=y_in.dtype).detach()
                else:
                    y_in = y_seq[:, k - 1, :].to(dtype=y_prev.dtype).detach()

            y_feat = y_in[:, self.gru_y_cols] if self.gru_y_cols is not None else y_in
            if y_transform == "sqrt":
                y_feat = y_feat.clamp_min(0.0).sqrt()
            elif y_transform == "sqrt_clamp1":
                y_feat = y_feat.clamp_min(0.0).sqrt().clamp_min(1.0)
            elif y_transform == "log1p":
                y_feat = torch.log1p(y_feat.clamp_min(0.0))

            x = self.lift(torch.cat([u_feat, y_feat], dim=-1))

            for i, (norm, layer) in enumerate(zip(self.norms, self.mamba_layers)):
                x_out, caches[i] = layer.step(norm(x), caches[i])
                x = x + x_out

            raw = self.head(x)

            if self.use_basal:
                raw_theta = raw[:, :self.theta_dim]
                if self.theta_bounded:
                    if self.theta_head_transform == "gamma":
                        theta_k = gamma(raw_theta, self.theta_lo_vec, self.theta_hi_vec)
                    else:
                        theta_k = log_gamma(raw_theta, self.theta_lo_vec, self.theta_hi_vec, tau=self.theta_head_tau)
                else:
                    theta_k = F.softplus(raw_theta)
                beta_k = raw[:, self.theta_dim:] * (y_prev / (y_prev + 1.0))
                beta_out[:, k, :] = beta_k
                y = y_prev + (u_k @ self.u_to_y_jump)
                y = self._rk4_substeps_basal(y, dt_k, theta_k, beta_k)
            else:
                if self.theta_bounded:
                    if self.theta_head_transform == "gamma":
                        theta_k = gamma(raw, self.theta_lo_vec, self.theta_hi_vec)
                    else:
                        theta_k = log_gamma(raw, self.theta_lo_vec, self.theta_hi_vec, tau=self.theta_head_tau)
                else:
                    theta_k = F.softplus(raw)
                if self._analytic_scaffold:
                    y = self.rhs.analytic_step(y_prev, dt_k, theta_k, analytic_ctx)
                else:
                    y = y_prev + (u_k @ self.u_to_y_jump)
                    y = self._rk4_substeps(y, dt_k, theta_k)

            y_out[:, k, :] = y
            th_out[:, k, :] = self.rhs.emit_theta(theta_k, y) if self._analytic_scaffold else theta_k
            y_prev = y

        return y_out, th_out, beta_out

    def _rk4_substeps(self, y, dt, theta):
        rhs = self.rhs
        n_sub = self.n_substeps
        dt = dt.unsqueeze(1)
        hdt = dt / float(n_sub)
        for _ in range(n_sub):
            k1 = rhs(y, theta)
            k2 = rhs(y + 0.5 * hdt * k1, theta)
            k3 = rhs(y + 0.5 * hdt * k2, theta)
            k4 = rhs(y + hdt * k3, theta)
            y = y + (hdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.clamp_min(y, 0.0)

    def _rk4_substeps_basal(self, y, dt, theta, beta):
        rhs = self.rhs
        n_sub = self.n_substeps
        dt = dt.unsqueeze(1)
        hdt = dt / float(n_sub)
        for _ in range(n_sub):
            k1 = rhs(y, theta) + beta
            k2 = rhs(y + 0.5 * hdt * k1, theta) + beta
            k3 = rhs(y + 0.5 * hdt * k2, theta) + beta
            k4 = rhs(y + hdt * k3, theta) + beta
            y = y + (hdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.clamp_min(y, 0.0)

    def _init_caches(self, B, device, dtype):
        caches = []
        cfg = self.mamba_layers[0].config
        for _ in range(len(self.mamba_layers)):
            h = None  # zero-initialised on first step by ssm_step
            inputs = torch.zeros(B, cfg.d_inner, cfg.d_conv - 1, device=device, dtype=dtype)
            caches.append((h, inputs))
        return caches
