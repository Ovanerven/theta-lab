"""
OdesLSTM: closed-loop mechanistic encoder using a hand-rolled sLSTM cell.

Why a custom cell instead of the xLSTM package's sLSTMLayer?
  - xLSTM's mLSTM blows memory because it materialises a (head_dim x head_dim)
    outer-product state plus QKV/conv buffers and (under the parallel kernel) a
    context-length chunk.
  - sLSTM is fundamentally a vanilla LSTM with exponential gates and an extra
    normaliser state, so its state shape is (B, H) per layer — same footprint
    as nn.LSTMCell. The xLSTM package wraps it in CUDA-kernel constraints that
    don't play nicely with our per-step Python loop.
  - The cell below implements eq. 8 of Beck et al. (2024) plus the log-space
    stabiliser (eq. 15). This is essential — naive exp() gates overflow within
    a few steps. Tanh on z, sigmoid on o, exp on i/f.

Drop-in replacement for OdeLSTM. Same forward signature, same kwargs.
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


class sLSTMCell(nn.Module):
    """Beck et al. sLSTM (eq. 8) with log-space stabiliser (eq. 15).

    State per cell: (h, c, n, m), each (B, H).
      h: hidden state
      c: cell state
      n: normaliser
      m: log-space stabiliser (max-trick) — keeps exp() bounded.
    """

    def __init__(self, input_size: int, hidden_size: int, forget_bias: float = 0.0):
        super().__init__()
        self.H = int(hidden_size)
        # i, f, z, o — input projection (W) and recurrent projection (R).
        self.W = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.R = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        if forget_bias != 0.0:
            with torch.no_grad():
                self.W.bias[hidden_size:2 * hidden_size].fill_(float(forget_bias))

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if state is None:
            B = x.shape[0]
            zero = x.new_zeros(B, self.H)
            h, c, n, m = zero, zero, zero, zero
        else:
            h, c, n, m = state

        gates = self.W(x) + self.R(h)
        i_pre, f_pre, z_pre, o_pre = gates.chunk(4, dim=-1)

        # Stabiliser: m_t = max(log f + m_{t-1}, log i). i and f are exp()'d,
        # so log i = i_pre, log f = f_pre.
        m_new = torch.maximum(f_pre + m, i_pre)
        i = torch.exp(i_pre - m_new)
        f = torch.exp(f_pre + m - m_new)
        z = torch.tanh(z_pre)
        o = torch.sigmoid(o_pre)

        c_new = f * c + i * z
        n_new = f * n + i
        h_new = o * (c_new / (n_new + 1e-6))
        return h_new, (h_new, c_new, n_new, m_new)


class _StackedSLSTM(nn.Module):
    """A stack of sLSTMCell layers with optional dropout between layers."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, forget_bias: float):
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        cells: List[sLSTMCell] = []
        for ell in range(self.num_layers):
            in_dim = input_size if ell == 0 else hidden_size
            cells.append(sLSTMCell(in_dim, hidden_size, forget_bias=forget_bias))
        self.cells = nn.ModuleList(cells)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]],
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]:
        # TorchScript can't widen `List[None]` to `List[Tuple[...]]`, so keep a
        # separately-typed Optional list as the per-layer input.
        states_in: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = []
        if states is None:
            for _ in range(self.num_layers):
                states_in.append(None)
        else:
            for s in states:
                states_in.append(s)
        new_states: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        h = x
        for ell, cell in enumerate(self.cells):
            h, st = cell(h, states_in[ell])
            new_states.append(st)
            if ell < self.num_layers - 1:
                h = self.drop(h)
        return h, new_states


class OdesLSTM(nn.Module):
    """
    Closed-loop mechanistic model with sLSTM encoder:
      (u_k, y_{k-1}) -> sLSTM -> theta_k
      y <- y + u_k @ jump
      y <- integrate ODE(scaffold, theta_k) over dt_k

    Mirrors OdeLSTM API: same constructor kwargs and forward signature.
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
        forget_bias_init: Optional[float] = None,
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
        self.slstm = _StackedSLSTM(
            input_size=enc_in,
            hidden_size=self.hidden,
            num_layers=int(num_layers),
            dropout=float(dropout),
            forget_bias=float(forget_bias_init) if forget_bias_init is not None else 0.0,
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

        states: Optional[List] = None
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
            z, states = self.slstm(x, states)
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
