from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaffolds import MechanisticScaffold
from models.u_features import u_feature_mult, build_u_enc


def gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    # Linear-sigmoid: arithmetic midpoint at x=0.
    return lo + (hi - lo) * torch.sigmoid(x)


def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    # Sigmoid in log-space: bounded in [lo, hi], geometric midpoint at x=0.
    # tau > 1 flattens the sigmoid (e.g. tau=2.3), preventing head saturation.
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x / tau))


class OdeLSTM(nn.Module):
    """
    Closed-loop mechanistic model:
      (u_k, y_{k-1}) -> lstm -> theta_k
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
        forget_bias_init: Optional[float] = None,
        legacy_forget_bias_bug: bool = False,
        gru_u_cols: Optional[list] = None,
        gru_y_cols: Optional[list] = None,
        lift_skip: bool = False,
        head_init: str = "default",  # "default" | "orthogonal" (xavier_ + zeros, unconditional)
        theta_head_transform: str = "log_gamma",   # "log_gamma" | "gamma"
        theta_head_tau: float = 1.0,
        u_transform: str = "none",   # encoder u-feature transform (sizes the lift)
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

        if theta_head_transform not in ("log_gamma", "gamma"):
            raise ValueError(f"theta_head_transform must be 'log_gamma' or 'gamma', got {theta_head_transform}")
        self.theta_head_transform = str(theta_head_transform)
        self.theta_head_tau = float(theta_head_tau)

        self._analytic_scaffold = bool(getattr(rhs, "has_analytic_step", False))
        self._tf_at_k_zero      = bool(getattr(rhs, "tf_at_k_zero", False))
        self.theta_dim_emit     = int(getattr(rhs, "theta_dim_emit", self.theta_dim))

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

        # Per-parameter bounds — use scaffold-defined if available, else broadcast scalar
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
        if self.lift_skip:
            self.lift = nn.Identity()
            lstm_input_dim = feat_in
        else:
            self.lift = nn.Sequential(
                nn.Linear(feat_in, lift_dim),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )
            lstm_input_dim = lift_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        head_out = self.theta_dim + self.P if self.use_basal else self.theta_dim
        self.head = nn.Linear(hidden, head_out)

        if head_init not in ("default", "orthogonal", "supervisor"):
            raise ValueError(f"head_init must be 'default' or 'orthogonal', got {head_init}")
        # "supervisor" is a backward-compatible alias for "orthogonal".
        if str(head_init) in ("orthogonal", "supervisor"):
            nn.init.xavier_uniform_(self.head.weight, gain=1.0)
            nn.init.zeros_(self.head.bias)

        # PyTorch LSTM bias layout per layer: [i, f, g, o], each block of size hidden.
        #
        # Three init modes:
        #   1. forget_bias_init=None           -> do nothing, keep PyTorch default
        #   2. forget_bias_init=1.0 (correct)  -> add_ to bias_ih only; preserves random
        #                                         init and gives an effective shift of
        #                                         exactly forget_bias_init (Gers/Jozefowicz).
        #   3. legacy_forget_bias_bug=True     -> reproduce the OLD buggy behavior:
        #                                         fill_ both bias_ih AND bias_hh forget
        #                                         blocks with forget_bias_init. This
        #                                         (a) destroys the random init and
        #                                         (b) doubles the effective bias because
        #                                             the two biases are summed inside the
        #                                             LSTM. Kept for A/B reproducibility
        #                                             of pre-fix runs.
        if legacy_forget_bias_bug:
            fb = 0.0 if forget_bias_init is None else float(forget_bias_init)
            n = hidden
            for name, p in self.lstm.named_parameters():
                if "bias" in name:  # matches both bias_ih_l* and bias_hh_l* (the bug)
                    with torch.no_grad():
                        p[n:2*n].fill_(fb)
        elif forget_bias_init is not None:
            n = hidden
            for name, p in self.lstm.named_parameters():
                if "bias_ih" in name:
                    with torch.no_grad():
                        p[n:2*n].add_(float(forget_bias_init))

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
        u_transform: str = "none",
        y_transform: str = "none",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        y_out    = torch.empty(B, K, self.P,                 device=y0.device, dtype=y0.dtype)
        th_out   = torch.empty(B, K, self.theta_dim_emit,    device=y0.device, dtype=y0.dtype)
        beta_out = torch.zeros(B, K, self.P,                 device=y0.device, dtype=y0.dtype)

        h = (
            torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=y0.device, dtype=y0.dtype),
            torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=y0.device, dtype=y0.dtype),
        )

        use_partial = obs_idx.numel() > 0

        # Analytic-scaffold context (e.g. dna_cum_total) and seeded initial state.
        analytic_ctx: dict = {}
        if self._analytic_scaffold:
            analytic_ctx = self.rhs.precompute_batch(y0, u_seq)
            y_prev = self.rhs.initial_state(y0)
        else:
            y_prev = y0

        # Encoder's view of u (gru_u_cols already applied); ODE jump uses raw delta.
        u_enc = build_u_enc(u_seq, dt_seq, self._u_transform, self.gru_u_idx, self._has_u_cols)

        for k in range(K):
            u_k    = u_seq[:, k, :]   # raw delta — used only for ODE jumps (non-analytic path)
            u_feat = u_enc[:, k, :]   # cols + transform already applied
            dt_k   = dt_seq[:, k]

            y_in = y_prev.detach()
            tf_fires = (k % tf_every == 0) if self._tf_at_k_zero else (k > 0 and k % tf_every == 0)
            if teacher_forcing and tf_fires and y_seq is not None:
                if use_partial:
                    y_in = y_prev.clone()
                    idx = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].to(dtype=y_in.dtype).detach()
                else:
                    y_in = y_seq[:, k - 1, :].to(dtype=y_prev.dtype).detach()

            # Encoder feature: optionally subset u/y columns and apply y_transform.
            y_feat = y_in[:, self.gru_y_cols] if self.gru_y_cols is not None else y_in
            if y_transform == "sqrt":
                y_feat = y_feat.clamp_min(0.0).sqrt()
            elif y_transform == "sqrt_clamp1":
                y_feat = y_feat.clamp_min(0.0).sqrt().clamp_min(1.0)
            elif y_transform == "log1p":
                y_feat = torch.log1p(y_feat.clamp_min(0.0))

            x = self.lift(torch.cat([u_feat, y_feat], dim=-1)).unsqueeze(1)
            z, h = self.lstm(x, h)
            raw = self.head(z.squeeze(1))

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

            y_out[:, k, :]  = y
            th_out[:, k, :] = self.rhs.emit_theta(theta_k, y) if self._analytic_scaffold else theta_k
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