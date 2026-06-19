"""
ODE-LMU: Legendre Memory Unit encoder for mechanistic ODE parameter inference.

Drop-in replacement for OdeRNN/OdeLSTM. Replaces the gated RNN with an LMU
(Voelker, Kajic & Eliasmith, NeurIPS 2019): a memory cell derived from d coupled
ODEs whose phase space is the Legendre-polynomial basis over a sliding window of
length `theta`. The memory matrices have eigenvalues on/near the unit circle, so
information is provably retained across long input-free intervals — exactly the
"early bolus, long observation tail" regime where GRU/LSTM forget.

Architecture per step (mirrors OdeRNN):
  (u_k, y_{k-1}) -> lift -> stacked LMU -> head -> theta_k
  y <- y + u_k @ jump
  y <- RK4(scaffold, theta_k, dt_k)

LMU cell (per layer):
  u_t = e_x x_t + e_h h_{t-1} + e_m m_{t-1}      (scalar memory input)
  m_t = A_bar m_{t-1} + B_bar u_t                (Legendre memory; ZOH-discretized)
  h_t = tanh(W_x x_t + W_h h_{t-1} + W_m m_t)    (hidden state)

A_bar/B_bar are the ZOH discretization of the continuous LegS system
  theta * m'(t) = A_c m(t) + B_c u(t),
computed once at construction (float64 for conditioning, stored as float32).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaffolds import MechanisticScaffold
from models.u_features import u_feature_mult, build_u_enc


def gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x / tau))


def _lmu_state_space(memory_size: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """ZOH-discretized LegS (Legendre) state-space matrices A_bar (d,d), B_bar (d,1).

    Continuous LegS system (normalized window theta):  theta * m' = A_c m + B_c u
      A_c[i,j] = (2i+1) * { -1            if i < j
                          { (-1)^(i-j+1)  if i >= j
      B_c[i]   = (2i+1) * (-1)^i
    Discretize at dt=1: A_bar = exp(A_c/theta), B_bar = A_c^{-1}(A_bar - I) B_c.
    """
    d = int(memory_size)
    idx = torch.arange(d, dtype=torch.long)
    R = (2.0 * idx.to(torch.float64) + 1.0)                      # (d,)
    i = idx.view(-1, 1)
    j = idx.view(1, -1)
    # (-1)^(i-j+1) without negative-base float pow (which is NaN in torch):
    sign_ge = torch.where(((i - j + 1) % 2) == 0,
                          torch.tensor(1.0, dtype=torch.float64),
                          torch.tensor(-1.0, dtype=torch.float64))
    A_c = R.view(-1, 1) * torch.where(i < j,
                                      torch.tensor(-1.0, dtype=torch.float64),
                                      sign_ge)                    # (d,d)
    sign_i = torch.where((idx % 2) == 0,
                         torch.tensor(1.0, dtype=torch.float64),
                         torch.tensor(-1.0, dtype=torch.float64))
    B_c = (R * sign_i).view(-1, 1)                                # (d,1)

    A_bar = torch.matrix_exp(A_c / float(theta))                 # (d,d)
    eye = torch.eye(d, dtype=torch.float64)
    B_bar = torch.linalg.solve(A_c, (A_bar - eye) @ B_c)         # (d,1)
    return A_bar.to(torch.float32), B_bar.to(torch.float32)


class LMUCell(nn.Module):
    """Single Legendre Memory Unit cell (memory + hidden state)."""

    def __init__(self, input_size: int, hidden_size: int, memory_size: int, theta: float):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.memory_size = int(memory_size)

        # Scalar memory-input encoders.
        self.e_x = nn.Linear(self.input_size, 1, bias=False)
        self.e_h = nn.Linear(self.hidden_size, 1, bias=False)
        self.e_m = nn.Linear(self.memory_size, 1, bias=False)
        # Hidden-state kernels.
        self.W_x = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_m = nn.Linear(self.memory_size, self.hidden_size, bias=False)

        A_bar, B_bar = _lmu_state_space(self.memory_size, theta)
        self.register_buffer("A_bar", A_bar)      # (d,d)
        self.register_buffer("B_bar", B_bar)      # (d,1)

        # Voelker-style init: Xavier on encoders/kernels, zero on e_m (so the
        # memory feedback starts off) and on W_h (clean start for the hidden map).
        nn.init.xavier_uniform_(self.e_x.weight)
        nn.init.xavier_uniform_(self.e_h.weight)
        nn.init.zeros_(self.e_m.weight)
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.zeros_(self.W_h.weight)
        nn.init.xavier_uniform_(self.W_m.weight)

    def forward(self, x: torch.Tensor, h: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self.e_x(x) + self.e_h(h) + self.e_m(m)              # (B,1)
        m = torch.matmul(m, self.A_bar.t()) + torch.matmul(u, self.B_bar.t())  # (B,d)
        h = torch.tanh(self.W_x(x) + self.W_h(h) + self.W_m(m))  # (B,hidden)
        return h, m


class OdeLMU(nn.Module):
    """
    Closed-loop mechanistic model with an LMU encoder:
      (u_k, y_{k-1}) -> LMU -> theta_k
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
        gru_u_cols: Optional[list] = None,
        gru_y_cols: Optional[list] = None,
        lift_skip: bool = False,
        head_init: str = "default",
        theta_head_transform: str = "log_gamma",
        theta_head_tau: float = 1.0,
        lmu_memory: int = 64,         # Legendre order d (memory state size)
        lmu_theta: float = 300.0,     # sliding-window length; set ≈ sequence length
        u_transform: str = "none",    # encoder u-feature transform (sizes the lift)
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
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.memory_size = int(lmu_memory)

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

        # Shared u-feature transform (channel-expanding modes size the lift here).
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
        if self.lift_skip:
            self.lift = nn.Identity()
            enc_in = feat_in
        else:
            self.lift = nn.Sequential(
                nn.Linear(feat_in, lift_dim),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )
            enc_in = lift_dim

        cells: List[nn.Module] = [LMUCell(enc_in, self.hidden, self.memory_size, lmu_theta)]
        for _ in range(self.num_layers - 1):
            cells.append(LMUCell(self.hidden, self.hidden, self.memory_size, lmu_theta))
        self.cells = nn.ModuleList(cells)
        self.drop = nn.Dropout(float(dropout))

        head_out = self.theta_dim + self.P if self.use_basal else self.theta_dim
        self.head = nn.Linear(self.hidden, head_out)
        if head_init not in ("default", "orthogonal", "supervisor"):
            raise ValueError(f"head_init must be 'default' or 'orthogonal', got {head_init}")
        # "supervisor" is a backward-compatible alias for "orthogonal".
        if str(head_init) in ("orthogonal", "supervisor"):
            nn.init.xavier_uniform_(self.head.weight, gain=1.0)
            nn.init.zeros_(self.head.bias)

        if u_to_y_jump.shape != (self.U, self.P):
            raise ValueError(f"u_to_y_jump must be (U,P)=({self.U},{self.P}), got {tuple(u_to_y_jump.shape)}")
        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

    def forward(
        self,
        y0: torch.Tensor,                     # (B,P)
        u_seq: torch.Tensor,                  # (B,K,U)
        dt_seq: torch.Tensor,                 # (B,K)
        obs_idx: torch.Tensor,                # (num_obs,)
        y_seq: Optional[torch.Tensor] = None, # (B,K,P) for teacher forcing
        teacher_forcing: bool = True,
        tf_every: int = 50,
        u_transform: str = "none",
        y_transform: str = "none",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        device, dtype = y0.device, y0.dtype
        y_out    = torch.empty(B, K, self.P,              device=device, dtype=dtype)
        th_out   = torch.empty(B, K, self.theta_dim_emit, device=device, dtype=dtype)
        beta_out = torch.zeros(B, K, self.P,              device=device, dtype=dtype)

        # Per-layer LMU state: hidden h and memory m, zero-initialised.
        hs: List[torch.Tensor] = [torch.zeros(B, self.hidden, device=device, dtype=dtype) for _ in range(self.num_layers)]
        ms: List[torch.Tensor] = [torch.zeros(B, self.memory_size, device=device, dtype=dtype) for _ in range(self.num_layers)]

        use_partial = obs_idx.numel() > 0

        analytic_ctx: dict = {}
        if self._analytic_scaffold:
            analytic_ctx = self.rhs.precompute_batch(y0, u_seq)
            y_prev = self.rhs.initial_state(y0)
        else:
            y_prev = y0

        # Encoder's view of u (gru_u_cols already applied); ODE jump uses raw delta.
        u_enc = build_u_enc(u_seq, dt_seq, self._u_transform, self.gru_u_idx, self._has_u_cols)

        for k in range(K):
            u_k     = u_seq[:, k, :]
            u_feat  = u_enc[:, k, :]   # cols + transform already applied
            dt_k    = dt_seq[:, k]

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
                y_feat = y_feat.clamp_min(1e-6).sqrt()
            elif y_transform == "sqrt_clamp1":
                y_feat = y_feat.clamp_min(1.0).sqrt()
            elif y_transform == "log1p":
                y_feat = torch.log1p(y_feat.clamp_min(0.0))

            x = self.lift(torch.cat([u_feat, y_feat], dim=-1))
            for li, cell in enumerate(self.cells):
                h_l, m_l = cell(x, hs[li], ms[li])
                hs[li] = h_l
                ms[li] = m_l
                x = self.drop(h_l) if self.training else h_l
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

            y_out[:, k, :]  = y
            th_out[:, k, :] = self.rhs.emit_theta(theta_k, y) if self._analytic_scaffold else theta_k
            y_prev = y

        return y_out, th_out, beta_out

    def _rk4_substeps(self, y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
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
            y  = y.clamp(0.0, 1e5)
        return y

    def _rk4_substeps_basal(self, y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
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
            y  = y.clamp(0.0, 1e5)
        return y
