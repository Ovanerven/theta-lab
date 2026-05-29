"""
OdeMinGRU: closed-loop mechanistic encoder using a hand-rolled minGRU cell.

minGRU (Feng et al., 2024 — "Were RNNs All We Needed?"):
  z_t      = sigmoid(W_z x_t)
  h~_t     = W_h x_t                 # candidate hidden — NO dependence on h_{t-1}
  h_t      = (1 - z_t) * h_{t-1} + z_t * h~_t

Key simplifications vs. classic GRU:
  - No reset gate.
  - Candidate h~_t depends only on x_t, not on h_{t-1}.
  - This makes the recurrence a *linear* convex combination, which the paper
    exploits for a parallel scan during training. Here we run it sequentially
    because the closed loop requires y_{k-1} from the ODE step before we can
    form x_k anyway — parallel scan is unavailable for closed-loop inference.

Drop-in mirror of OdeLSTM/OdeRNN: same constructor kwargs and forward signature.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaffolds import MechanisticScaffold


def gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x / tau))


class minGRUCell(nn.Module):
    """Sequential-mode minGRU cell. State shape (B, H)."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.H = int(hidden_size)
        # Pack [z | h~] in one matmul.
        self.lin = nn.Linear(input_size, 2 * hidden_size, bias=True)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        if h is None:
            h = x.new_zeros(x.shape[0], self.H)
        z_pre, h_tilde = self.lin(x).chunk(2, dim=-1)
        z = torch.sigmoid(z_pre)
        return (1.0 - z) * h + z * h_tilde


class _StackedMinGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        cells: List[minGRUCell] = []
        for ell in range(self.num_layers):
            in_dim = input_size if ell == 0 else hidden_size
            cells.append(minGRUCell(in_dim, hidden_size))
        self.cells = nn.ModuleList(cells)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, states: Optional[List[torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if states is None:
            states = [None] * self.num_layers  # type: ignore[list-item]
        new_states: List[torch.Tensor] = []
        h = x
        for ell, cell in enumerate(self.cells):
            h = cell(h, states[ell])
            new_states.append(h)
            if ell < self.num_layers - 1:
                h = self.drop(h)
        return h, new_states


class OdeMinGRU(nn.Module):
    """
    Closed-loop mechanistic model with minGRU encoder.

    (u_k, y_{k-1}) -> minGRU stack -> theta_k
    y <- y + u_k @ jump   (or analytic_step for analytic scaffolds)
    y <- integrate ODE(scaffold, theta_k) over dt_k
    """

    def __init__(
        self,
        *,
        U: int,
        rhs: MechanisticScaffold,
        u_to_y_jump: torch.Tensor,
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

        if rhs.theta_lo_vec is not None and rhs.theta_hi_vec is not None:
            lo = torch.tensor(rhs.theta_lo_vec, dtype=torch.float32)
            hi = torch.tensor(rhs.theta_hi_vec, dtype=torch.float32)
        else:
            lo = torch.full((self.theta_dim,), theta_lo)
            hi = torch.full((self.theta_dim,), theta_hi)
        self.register_buffer("theta_lo_vec", lo)
        self.register_buffer("theta_hi_vec", hi)

        self.lift_skip = bool(lift_skip)
        feat_in = u_cols_dim + y_cols_dim
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

        self.hidden = int(hidden)
        self.mingru = _StackedMinGRU(
            input_size=enc_in,
            hidden_size=self.hidden,
            num_layers=int(num_layers),
            dropout=float(dropout),
        )

        head_out = self.theta_dim + self.P if self.use_basal else self.theta_dim
        self.head = nn.Linear(self.hidden, head_out)
        if head_init not in ("default", "supervisor"):
            raise ValueError(f"head_init must be 'default' or 'supervisor', got {head_init}")
        if str(head_init) == "supervisor":
            nn.init.xavier_uniform_(self.head.weight, gain=1.0)
            nn.init.zeros_(self.head.bias)

        if u_to_y_jump.shape != (self.U, self.P):
            raise ValueError(f"u_to_y_jump must be (U,P)=({self.U},{self.P}), got {tuple(u_to_y_jump.shape)}")
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
        y_out = torch.empty(B, K, self.P, device=y0.device, dtype=y0.dtype)
        th_out = torch.empty(B, K, self.theta_dim_emit, device=y0.device, dtype=y0.dtype)
        beta_out = torch.zeros(B, K, self.P, device=y0.device, dtype=y0.dtype)

        states: Optional[List[torch.Tensor]] = None
        use_partial = obs_idx.numel() > 0

        analytic_ctx: dict = {}
        if self._analytic_scaffold:
            analytic_ctx = self.rhs.precompute_batch(y0, u_seq)
            y_prev = self.rhs.initial_state(y0)
        else:
            y_prev = y0

        if u_transform == "cumsum" or u_transform == "cumsum_sqrt":
            u_enc = u_seq.cumsum(dim=1)
        else:
            u_enc = u_seq
        if u_transform == "sqrt" or u_transform == "cumsum_sqrt":
            u_enc = u_enc.clamp_min(0.0).sqrt()

        for k in range(K):
            u_k = u_seq[:, k, :]
            u_enc_k = u_enc[:, k, :]
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

            u_feat = u_enc_k[:, self.gru_u_cols] if self.gru_u_cols is not None else u_enc_k
            y_feat = y_in[:, self.gru_y_cols] if self.gru_y_cols is not None else y_in
            if y_transform == "sqrt":
                y_feat = y_feat.clamp_min(0.0).sqrt()
            elif y_transform == "sqrt_clamp1":
                y_feat = y_feat.clamp_min(0.0).sqrt().clamp_min(1.0)
            elif y_transform == "log1p":
                y_feat = torch.log1p(y_feat.clamp_min(0.0))

            x = self.lift(torch.cat([u_feat, y_feat], dim=-1))
            z, states = self.mingru(x, states)
            raw = self.head(z)

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
