"""OdeRNN variant with a stateless-residual ("basal v2") head.

Differences vs. OdeRNN:
  * The basal/correction term is produced by a SEPARATE MLP whose input is
    (u_k transformed, theta_{k-1}, y_{k-1} on a restricted obs slice).
    It does NOT consume the GRU hidden state, and it is NOT a function of
    the latent state trajectory — which fits the real-data setting where
    only inputs and a couple of endpoint observables are tracked.
  * Forward takes a `basal_scale: float` argument (0 disables the basal,
    1 lets it act at full strength). The training loop drives this through
    a curriculum (theta-only first, then ramp basal in).
  * Basal head is zero-initialised so the model starts identical to the
    no-basal OdeRNN at scale=1, modulo the head wiring.

Everything else (theta head, scaffolds, RK4, analytic-step bypass, jumps,
y0_theta_init, transforms, GRU variants/inits, head_bottle, …) mirrors
OdeRNN so the two are diff-comparable.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaffolds import MechanisticScaffold
from models.ode_rnn import gamma, log_gamma, StackedGRUCellBlock


class OdeRNNBasalV2(nn.Module):
    __constants__ = [
        "_analytic_scaffold", "_tf_at_k_zero", "theta_dim_emit",
        "_has_gru_u_cols", "_has_gru_y_cols",
        "_has_basal_y_cols",
    ]

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
        theta_bounded: bool = True,
        gru_u_cols: Optional[List[int]] = None,
        gru_y_cols: Optional[List[int]] = None,
        basal_y_cols: Optional[List[int]] = None,    # y columns the basal MLP sees; None = reuse gru_y_cols
        basal_hidden: int = 64,
        basal_layers: int = 2,
        basal_max_amplitude: float | List[float] = 0.1,  # per-time-unit cap on |beta|.
                                                         # Scalar broadcasts across all P species; a list of length P
                                                         # sets per-species caps (recommended when y scales differ wildly,
                                                         # e.g. proteins ~10^3 vs DNA ~1). The contribution to state per
                                                         # step is bounded by basal_max_amplitude * dt.
        head_bias_init: float = 0.0,
        head_bias_init_vec: Optional[List[float]] = None,
        head_weight_gain: float = 1.0,
        detach_y_prev: bool = True,
        detach_theta_prev: bool = True,              # detach theta_{k-1} into the basal MLP
        basal_use_hidden: bool = True,               # feed GRU hidden state z_k into the basal MLP.
                                                     # Gives the basal the full u/y_obs history (encoded by the GRU)
                                                     # for free, while still being state-trajectory-independent.
        detach_hidden_basal: bool = True,            # detach z_k before the basal MLP so basal training never
                                                     # back-props gradients into the theta encoder.
        u_minmax_max: Optional[torch.Tensor] = None,
        u_minmax_cols: Optional[List[int]] = None,
        theta_head_transform: str = "log_gamma",
        theta_head_tau: float = 1.0,
        head_bottle: bool = False,
        head_bottle_dims: Optional[List[int]] = None,
        lift_skip: bool = False,
        gru_variant: str = "nn_gru",
        gru_init: str = "default",
        head_init: str = "default",
        y0_theta_init: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.U = int(U)
        self.P = int(rhs.P)
        self.theta_dim = int(rhs.theta_dim)
        self.rhs = rhs
        self._analytic_scaffold = bool(getattr(rhs, "has_analytic_step", False))
        self._tf_at_k_zero      = bool(getattr(rhs, "tf_at_k_zero", False))
        self.theta_dim_emit     = int(getattr(rhs, "theta_dim_emit", self.theta_dim))
        self.n_substeps   = int(n_substeps)
        self.theta_lo     = float(theta_lo)
        self.theta_hi     = float(theta_hi)
        self.theta_bounded = bool(theta_bounded)
        self.detach_y_prev = bool(detach_y_prev)
        self.detach_theta_prev = bool(detach_theta_prev)
        self.basal_use_hidden = bool(basal_use_hidden)
        self.detach_hidden_basal = bool(detach_hidden_basal)
        if theta_head_transform not in ("log_gamma", "gamma"):
            raise ValueError(f"theta_head_transform must be 'log_gamma' or 'gamma', got {theta_head_transform}")
        self.theta_head_transform = str(theta_head_transform)
        self.theta_head_tau = float(theta_head_tau)
        self.head_bottle_enabled = bool(head_bottle)
        self.head_bottle_dims = list(head_bottle_dims) if head_bottle_dims is not None else [120, 40]
        self.lift_skip = bool(lift_skip)
        if gru_variant not in ("nn_gru", "stacked_cell"):
            raise ValueError(f"gru_variant must be 'nn_gru' or 'stacked_cell', got {gru_variant}")
        if gru_init not in ("default", "supervisor"):
            raise ValueError(f"gru_init must be 'default' or 'supervisor', got {gru_init}")
        if head_init not in ("default", "supervisor"):
            raise ValueError(f"head_init must be 'default' or 'supervisor', got {head_init}")
        self.gru_variant = str(gru_variant)
        self.gru_init = str(gru_init)
        self.head_init = str(head_init)
        self.y0_theta_init = bool(y0_theta_init)

        if rhs.theta_lo_vec is not None and rhs.theta_hi_vec is not None:
            lo = torch.tensor(rhs.theta_lo_vec, dtype=torch.float32)
            hi = torch.tensor(rhs.theta_hi_vec, dtype=torch.float32)
        else:
            lo = torch.full((self.theta_dim,), theta_lo)
            hi = torch.full((self.theta_dim,), theta_hi)
        self.register_buffer("theta_lo_vec", lo)
        self.register_buffer("theta_hi_vec", hi)

        # Column subsets ---------------------------------------------------
        self.gru_u_cols = list(gru_u_cols) if gru_u_cols is not None else None
        self._has_gru_u_cols: bool = gru_u_cols is not None
        self.register_buffer(
            "gru_u_idx",
            torch.tensor(list(gru_u_cols) if gru_u_cols is not None else [], dtype=torch.long),
            persistent=False,
        )
        self.gru_y_cols = list(gru_y_cols) if gru_y_cols is not None else None
        self._has_gru_y_cols: bool = gru_y_cols is not None
        self.register_buffer(
            "gru_y_idx",
            torch.tensor(list(gru_y_cols) if gru_y_cols is not None else [], dtype=torch.long),
            persistent=False,
        )
        gru_y_dim = len(self.gru_y_cols) if self.gru_y_cols is not None else self.P
        gru_feat_dim = len(self.gru_u_cols) if self.gru_u_cols is not None else self.U

        # basal_y_cols defaults to gru_y_cols so the basal sees the same observable slice.
        if basal_y_cols is not None:
            self.basal_y_cols = list(basal_y_cols)
            self._has_basal_y_cols = True
            basal_y_idx_t = torch.tensor(list(basal_y_cols), dtype=torch.long)
            basal_y_dim = len(self.basal_y_cols)
        else:
            self.basal_y_cols = self.gru_y_cols
            self._has_basal_y_cols = self._has_gru_y_cols
            basal_y_idx_t = self.gru_y_idx.clone() if self._has_gru_y_cols else torch.zeros(0, dtype=torch.long)
            basal_y_dim = gru_y_dim
        self.register_buffer("basal_y_idx", basal_y_idx_t, persistent=False)

        # ------------------------------------------------------------------
        feat_in = gru_feat_dim + gru_y_dim
        if self.lift_skip:
            self.lift = nn.Identity()
            gru_input_dim = feat_in
        else:
            self.lift = nn.Sequential(
                nn.Linear(feat_in, lift_dim),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )
            gru_input_dim = lift_dim
        if self.gru_variant == "stacked_cell":
            self.gru = StackedGRUCellBlock(gru_input_dim, hidden, num_layers, dropout)
        else:
            self.gru = nn.GRU(
                input_size=gru_input_dim,
                hidden_size=hidden,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        if self.head_bottle_enabled:
            dims = self.head_bottle_dims
            layers: list[nn.Module] = []
            prev = hidden
            for d in dims:
                layers += [nn.Linear(prev, int(d)), nn.SiLU()]
                prev = int(d)
            self.head_bottle = nn.Sequential(*layers)
            head_in = prev
        else:
            self.head_bottle = nn.Identity()
            head_in = hidden
        self.head = nn.Linear(head_in, self.theta_dim)

        # Theta-head init (mirrors OdeRNN logic).
        if self.head_init == "supervisor":
            nn.init.xavier_uniform_(self.head.weight, gain=1.0)
            nn.init.zeros_(self.head.bias)
        else:
            if head_bias_init != 0.0:
                nn.init.constant_(self.head.bias, float(head_bias_init))
            if head_weight_gain != 1.0:
                nn.init.xavier_uniform_(self.head.weight, gain=float(head_weight_gain))
            if head_bias_init_vec is not None:
                vec = torch.as_tensor(list(head_bias_init_vec), dtype=self.head.bias.dtype)
                if vec.numel() != self.theta_dim:
                    raise ValueError(
                        f"head_bias_init_vec has length {vec.numel()}, expected theta_dim={self.theta_dim}"
                    )
                with torch.no_grad():
                    self.head.bias[:self.theta_dim].copy_(vec)

        if self.gru_init == "supervisor":
            if self.gru_variant == "stacked_cell":
                for c in self.gru.cells:
                    nn.init.orthogonal_(c.weight_hh)
                    nn.init.xavier_uniform_(c.weight_ih)
                    nn.init.zeros_(c.bias_hh)
                    nn.init.zeros_(c.bias_ih)
            else:
                for name, p in self.gru.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(p)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

        if self.y0_theta_init:
            self.y0_mlp = nn.Sequential(
                nn.Linear(gru_y_dim, lift_dim),
                nn.SiLU(),
                nn.Linear(lift_dim, self.theta_dim),
            )
            nn.init.zeros_(self.y0_mlp[-1].weight)
            nn.init.zeros_(self.y0_mlp[-1].bias)
        else:
            self.y0_mlp = None

        # Basal MLP --------------------------------------------------------
        # Input: [u_k transformed, theta_prev (bounded), y_prev on basal_y slice transformed,
        #         (optional) GRU hidden state z_k of size `hidden`].
        basal_in = gru_feat_dim + self.theta_dim + basal_y_dim
        if self.basal_use_hidden:
            basal_in += int(hidden)
        b_layers: list[nn.Module] = []
        prev = basal_in
        for _ in range(max(1, int(basal_layers) - 1)):
            b_layers += [nn.Linear(prev, int(basal_hidden)), nn.SiLU()]
            prev = int(basal_hidden)
        b_layers.append(nn.Linear(prev, self.P))
        self.basal_mlp = nn.Sequential(*b_layers)
        # Zero-init the final layer so beta=0 at start; the curriculum is what
        # eventually lets the basal contribute.
        nn.init.zeros_(self.basal_mlp[-1].weight)
        nn.init.zeros_(self.basal_mlp[-1].bias)

        # Per-species max amplitude on β (after tanh squash). Stored as a
        # (P,) buffer so it broadcasts cleanly against (B, P) tensors and
        # moves with the model to GPU.
        if isinstance(basal_max_amplitude, (list, tuple)):
            amp_t = torch.as_tensor(list(basal_max_amplitude), dtype=torch.float32)
            if amp_t.numel() != self.P:
                raise ValueError(
                    f"basal_max_amplitude list has length {amp_t.numel()}, expected P={self.P}"
                )
        else:
            amp_t = torch.full((self.P,), float(basal_max_amplitude), dtype=torch.float32)
        self.register_buffer("_basal_amp", amp_t, persistent=True)

        if u_to_y_jump.shape != (self.U, self.P):
            raise ValueError(f"u_to_y_jump must be (U,P)=({self.U},{self.P}), got {tuple(u_to_y_jump.shape)}")
        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

        u_max_full = torch.ones(self.U, dtype=torch.float32)
        if u_minmax_max is not None and u_minmax_cols is not None:
            cols = torch.as_tensor(u_minmax_cols, dtype=torch.long)
            u_max_full[cols] = u_minmax_max.float().clamp_min(1e-8)
            self._has_u_minmax = True
        else:
            self._has_u_minmax = False
        self.register_buffer("u_minmax_max_full", u_max_full, persistent=False)

    # -----------------------------------------------------------------
    def _apply_y_transform(self, y_feat: torch.Tensor, y_transform: str) -> torch.Tensor:
        if y_transform == "sqrt":
            return y_feat.clamp_min(0.0).sqrt()
        if y_transform == "sqrt_clamp1":
            return y_feat.clamp_min(0.0).sqrt().clamp_min(1.0)
        if y_transform == "log1p":
            return torch.log1p(y_feat.clamp_min(0.0))
        return y_feat

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
        basal_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        y_out    = torch.empty(B, K, self.P, device=y0.device, dtype=y0.dtype)
        th_out   = torch.empty(B, K, self.theta_dim_emit, device=y0.device, dtype=y0.dtype)
        beta_out = torch.zeros(B, K, self.P, device=y0.device, dtype=y0.dtype)

        analytic_ctx: Dict[str, torch.Tensor] = {}
        if self._analytic_scaffold:
            analytic_ctx = self.rhs.precompute_batch(y0, u_seq)

        h = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size, device=y0.device, dtype=y0.dtype)
        use_partial = obs_idx.numel() > 0

        if u_transform == "cumsum" or u_transform == "cumsum_sqrt":
            u_gru = u_seq.cumsum(dim=1)
        else:
            u_gru = u_seq
        if u_transform == "minmax" or u_transform == "minmax_sqrt":
            if not self._has_u_minmax:
                raise ValueError(
                    "u_transform=" + str(u_transform) + " requires u_minmax_max/u_minmax_cols at model init."
                )
            u_gru = u_gru / self.u_minmax_max_full.view(1, 1, -1)
        if u_transform == "sqrt" or u_transform == "cumsum_sqrt" or u_transform == "minmax_sqrt":
            u_gru = u_gru.clamp_min(0.0).sqrt()

        if self.y0_mlp is not None:
            y0_feat = torch.index_select(y0, dim=1, index=self.gru_y_idx) if self._has_gru_y_cols else y0
            y0_feat = self._apply_y_transform(y0_feat, y_transform)
            raw_y0_bias: Optional[torch.Tensor] = self.y0_mlp(y0_feat)
        else:
            raw_y0_bias = None

        y_prev = self.rhs.initial_state(y0) if self._analytic_scaffold else y0
        # theta_prev starts as the geometric/arithmetic midpoint of the bounds.
        if self.theta_bounded:
            if self.theta_head_transform == "gamma":
                theta_prev = 0.5 * (self.theta_lo_vec + self.theta_hi_vec).unsqueeze(0).expand(B, -1)
            else:
                theta_prev = torch.sqrt(self.theta_lo_vec * self.theta_hi_vec).unsqueeze(0).expand(B, -1)
        else:
            theta_prev = torch.zeros(B, self.theta_dim, device=y0.device, dtype=y0.dtype)
        theta_prev = theta_prev.contiguous()

        for k in range(K):
            u_k     = u_seq[:, k, :]
            u_gru_k = u_gru[:, k, :]
            dt_k    = dt_seq[:, k]

            y_in = y_prev.detach() if self.detach_y_prev else y_prev

            tf_fires = (k % tf_every == 0) if self._tf_at_k_zero else (k > 0 and k % tf_every == 0)
            if teacher_forcing and tf_fires and y_seq is not None:
                if use_partial:
                    y_in = y_prev.clone()
                    idx = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].to(dtype=y_in.dtype).detach()
                else:
                    y_in = y_seq[:, k - 1, :].to(dtype=y_prev.dtype).detach()

            u_gru_k_feat = torch.index_select(u_gru_k, dim=1, index=self.gru_u_idx) if self._has_gru_u_cols else u_gru_k
            y_in_feat = torch.index_select(y_in, dim=1, index=self.gru_y_idx) if self._has_gru_y_cols else y_in
            y_in_feat = self._apply_y_transform(y_in_feat, y_transform)
            feat = torch.cat([u_gru_k_feat, y_in_feat], dim=-1)
            x = self.lift(feat).unsqueeze(1)
            z, h = self.gru(x, h)
            raw_theta = self.head(self.head_bottle(z.squeeze(1)))
            if raw_y0_bias is not None:
                raw_theta = raw_theta + raw_y0_bias

            if self.theta_bounded:
                if self.theta_head_transform == "gamma":
                    theta_k = gamma(raw_theta, self.theta_lo_vec, self.theta_hi_vec)
                else:
                    theta_k = log_gamma(raw_theta, self.theta_lo_vec, self.theta_hi_vec, tau=self.theta_head_tau)
            else:
                theta_k = F.softplus(raw_theta)

            # --- basal branch ---
            # Inputs: u_k transformed (basal sees the same view as the GRU),
            # theta_prev (bounded, optionally detached), y_prev on basal_y slice.
            y_in_basal = torch.index_select(y_in, dim=1, index=self.basal_y_idx) if self._has_basal_y_cols else y_in
            y_in_basal = self._apply_y_transform(y_in_basal, y_transform)
            theta_b = theta_prev.detach() if self.detach_theta_prev else theta_prev
            basal_parts = [u_gru_k_feat, theta_b, y_in_basal]
            if self.basal_use_hidden:
                # z is (B, 1, hidden) from this step's GRU update — it summarises
                # the full u/y_obs history seen so far.
                z_flat = z.squeeze(1)
                if self.detach_hidden_basal:
                    z_flat = z_flat.detach()
                basal_parts.append(z_flat)
            basal_feat = torch.cat(basal_parts, dim=-1)
            beta_raw = self.basal_mlp(basal_feat)

            # Bound β with tanh and a per-species max amplitude so its magnitude
            # is intrinsically capped regardless of Adam's first step or the
            # trajectory time horizon. Without this β can grow large within one
            # update (zero-init → Adam first step is ~±lr per param), and since
            # β enters the dynamics as ∫β dt, even a small β·dt can wipe out
            # the state over long horizons.
            #
            # Interpretation of the cap: β_max = max_amp_per_unit_time.
            # The state change contribution per step is β·dt; over the whole
            # trajectory of length T it is at most max_amp · T per species.
            beta_bounded = self._basal_amp * torch.tanh(beta_raw)

            # Magnitude penalty is computed on the BOUNDED (unscaled-by-ramp)
            # β so it doesn't get arbitrarily weakened during the ramp window.
            beta_out[:, k, :] = beta_bounded

            # `basal_scale` is the curriculum ramp gate (0 during warmup → 1
            # post-ramp). Apply it only to the value that actually enters the
            # dynamics, not to the penalty term.
            beta_used = beta_bounded * float(basal_scale) if basal_scale != 1.0 else beta_bounded

            if self._analytic_scaffold:
                y = self.rhs.analytic_step(y_prev, dt_k, theta_k, analytic_ctx)
                y = y + beta_used * dt_k.unsqueeze(1)
                y = torch.clamp_min(y, 0.0)
            else:
                y = y_prev + (u_k @ self.u_to_y_jump)
                y = self._rk4_substeps_basal(y, dt_k, theta_k, beta_used)

            y_out[:, k, :] = y
            th_out[:, k, :] = self.rhs.emit_theta(theta_k, y) if self._analytic_scaffold else theta_k
            y_prev = y
            theta_prev = theta_k

        return y_out, th_out, beta_out

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

    # --- training-loop helpers ---
    def theta_parameters(self):
        """Params that produce theta (GRU + lift + head + bottle + y0 MLP)."""
        modules = [self.lift, self.gru, self.head, self.head_bottle]
        if self.y0_mlp is not None:
            modules.append(self.y0_mlp)
        for m in modules:
            for p in m.parameters():
                yield p

    def basal_parameters(self):
        for p in self.basal_mlp.parameters():
            yield p
