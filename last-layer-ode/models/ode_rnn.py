from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaffolds import MechanisticScaffold


def gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    # Linear-sigmoid: arithmetic midpoint at x=0. Matches supervisor reference.
    return lo + (hi - lo) * torch.sigmoid(x)


def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    # Sigmoid in log-space: bounded in [lo, hi], geometric midpoint at x=0.
    # tau > 1 flattens the sigmoid (supervisor uses tau=2.3), preventing head saturation.
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x / tau))


class StackedGRUCellBlock(nn.Module):
    """Drop-in replacement for nn.GRU that mirrors the supervisor's stacked GRUCell.

    Differs from nn.GRU: dropout is applied to EVERY layer's output (including
    the last) when training, so the head sees a dropped activation.
    Default init: orthogonal_(W_hh) + xavier_uniform_(W_ih) + zeros for biases.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        cells = [nn.GRUCell(self.input_size, self.hidden_size)]
        cells += [nn.GRUCell(self.hidden_size, self.hidden_size) for _ in range(self.num_layers - 1)]
        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout(float(dropout))
        for c in self.cells:
            nn.init.orthogonal_(c.weight_hh)
            nn.init.xavier_uniform_(c.weight_ih)
            nn.init.zeros_(c.bias_hh)
            nn.init.zeros_(c.bias_ih)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        # x: (B, 1, input_size); h: (num_layers, B, hidden_size)
        out = x.squeeze(1)
        new_h = []
        for li, cell in enumerate(self.cells):
            state = cell(out, h[li])
            new_h.append(state)
            out = self.dropout(state) if self.training else state
        h_new = torch.stack(new_h, dim=0)
        return out.unsqueeze(1), h_new


class OdeRNN(nn.Module):
    """
    Closed-loop mechanistic model:
      (u_k, y_{k-1}) -> GRU -> theta_k
      y <- y + u_k @ jump
      y <- integrate ODE(scaffold, theta_k) over dt_k
    """

    # Compile-time flags for TorchScript so the analytic-scaffold branch is DCE'd
    # for non-analytic scaffolds, and so the TF-at-k=0 branch evaluates statically.
    # `_has_gru_*_cols` flags route the encoder feature build to either index_select
    # (TorchScript-friendly) or full-passthrough — selected once at script time.
    __constants__ = [
        "_analytic_scaffold", "_tf_at_k_zero", "theta_dim_emit",
        "_has_gru_u_cols", "_has_gru_y_cols", "encoder_use_time", "encoder_use_log_dt",
        "u_transform", "_u_preselected", "rk4_residual",
    ]

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
        gru_y_cols: Optional[list] = None,  # which y columns enter the GRU (None = all P).
                                            # Restricting to obs_idx prevents the GRU from being
                                            # confused by unobservable latent states.
        head_bias_init: float = 0.0,        # init all head biases to this value (<0 starts theta near lo)
        head_bias_init_vec: Optional[list] = None,  # per-theta-component bias init vector of length theta_dim;
                                                    # when provided, overrides `head_bias_init` on the theta slice
                                                    # of head.bias. Computed offline from per-step fit medians,
                                                    # inverse-transformed through the active theta_head_transform.
        head_weight_gain: float = 1.0,      # Xavier gain for head weights (>1 amplifies per-experiment variation)
        detach_y_prev: bool = True,         # if False, allow gradients to flow through y_prev in GRU
        u_minmax_max: Optional[torch.Tensor] = None,  # per-channel max for "minmax" / "minmax_sqrt" u_transform
        u_minmax_cols: Optional[list] = None,         # indices in u_seq[:,:,U] that u_minmax_max corresponds to
        theta_head_transform: str = "log_gamma",      # "log_gamma" (default) | "gamma" (linear-sigmoid; supervisor)
        theta_head_tau: float = 1.0,                  # temperature for log_gamma sigmoid; supervisor uses 2.3 (gentler slope)
        head_bottle: bool = False,                    # if True, insert hidden→…→SiLU stack before head (see head_bottle_dims)
        head_bottle_dims: Optional[list] = None,      # bottle layer widths; default [120, 40]; supervisor: [128, 64]
        lift_skip: bool = False,                      # if True, drop the lift MLP and feed feat→GRU directly (supervisor)
        gru_variant: str = "nn_gru",                  # "nn_gru" (default) | "stacked_cell" (supervisor's stacked GRUCell + dropout-on-last)
        gru_init: str = "default",                    # "default" (PyTorch defaults) | "supervisor" (orthogonal_ W_hh + xavier_ W_ih + zeros)
        head_init: str = "default",                   # "default" (PyTorch defaults; respect head_bias_init/head_weight_gain) | "supervisor" (xavier_ + zeros, unconditional)
        y0_theta_init: bool = False,                  # if True, add an MLP(y0) logit bias so the GRU starts from a
                                                      # per-sample theta prior rather than from the population mean
        encoder_use_time: bool = False,               # if True, concat τ_k = k/(K-1) ∈ [0,1] to the encoder
        encoder_use_log_dt: bool = False,             # if True, concat log(dt_k) to the encoder (dt-awareness for variable grids)
                                                      # feature vector (Experiment A in new_scaffolds.tex §3.1).
        u_transform: str = "none",                    # encoder u-feature transform. Channel-EXPANDING modes must be known
                                                      # at init so the lift layer is sized correctly:
                                                      #   "pulse_cumsum_sqrt"     → [sqrt(pulse), sqrt(cumsum)]    (2x cols)
                                                      #   "cumsum_timesince_sqrt" → [sqrt(cumsum), log1p(t_since)] (2x cols)
                                                      #   "decay_trace"           → 3 dt-aware leaky integrators   (3x cols)
                                                      # all other modes ("none"/"sqrt"/"cumsum"/"cumsum_sqrt"/minmax) stay 1x.
        u_decay_taus: Optional[list] = None,          # leaky-integrator time constants (seconds) for "decay_trace"; default fast/med/slow
        rk4_residual: bool = False,                   # idea #1: add a state-dependent neural residual g(y) to the RHS,
                                                      # re-evaluated at EVERY RK4 stage (true UDE term), not step-constant
                                                      # like the basal beta. Zero-init so training starts as pure mechanism.
        rk4_residual_hidden: int = 64,                # width of the residual MLP
        rk4_residual_layers: int = 2,                 # number of hidden layers in the residual MLP
        **kwargs,
    ):
        super().__init__()
        self.U = int(U)
        self.P = int(rhs.P)
        self.theta_dim = int(rhs.theta_dim)
        self.rhs = rhs
        # Analytic-scaffold hooks: when the scaffold defines a closed-form step
        # (e.g. IvttAnalyticScaffold), bypass the u-jump + RK4 path and let the
        # scaffold own integration. Default-False keeps every other scaffold's
        # codepath unchanged.
        self._analytic_scaffold = bool(getattr(rhs, "has_analytic_step", False))
        self._tf_at_k_zero      = bool(getattr(rhs, "tf_at_k_zero", False))
        self.theta_dim_emit     = int(getattr(rhs, "theta_dim_emit", self.theta_dim))
        self.n_substeps   = int(n_substeps)
        self.use_basal    = bool(use_basal)
        self.theta_lo     = float(theta_lo)
        self.theta_hi     = float(theta_hi)
        self.theta_bounded = bool(theta_bounded)
        # gru_u_cols / gru_y_cols storage — keep the original `Optional[List[int]]`
        # shadow attrs for back-compat (some external code may inspect them) but
        # the forward loop uses TorchScript-friendly buffer + bool flag below.
        self.gru_u_cols   = list(gru_u_cols) if gru_u_cols is not None else None
        self._has_gru_u_cols: bool = gru_u_cols is not None
        self.register_buffer(
            "gru_u_idx",
            torch.tensor(list(gru_u_cols) if gru_u_cols is not None else [], dtype=torch.long),
            persistent=False,
        )
        self.detach_y_prev = bool(detach_y_prev)
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
        # ── encoder u-feature transform ──────────────────────────────────────
        # Channel-expanding modes carry persistent magnitude AND timing/recency.
        # The multiplier sizes the lift layer; the per-step build lives in forward.
        self.u_transform = str(u_transform)
        if self.u_transform == "pulse_cumsum_sqrt" or self.u_transform == "cumsum_timesince_sqrt":
            self._u_feat_mult = 2
        elif (self.u_transform == "decay_trace" or self.u_transform == "pulse_cumsum_timesince"
              or self.u_transform == "pulse_cumsum_static"):
            self._u_feat_mult = 3
        else:
            self._u_feat_mult = 1
        # These modes pre-select gru_u_cols ONCE (loop-invariant) then expand, so
        # the per-step index_select in the forward loop is skipped for them.
        self._u_preselected = (self._u_feat_mult > 1)
        _taus = list(u_decay_taus) if u_decay_taus is not None else [300.0, 3600.0, 36000.0]
        self.register_buffer("u_decay_taus", torch.tensor(_taus, dtype=torch.float32), persistent=False)
        self.y0_theta_init = bool(y0_theta_init)
        self.encoder_use_time = bool(encoder_use_time)
        self.encoder_use_log_dt = bool(encoder_use_log_dt)

        # Per-parameter bounds — use scaffold-defined if available, else broadcast scalar
        if rhs.theta_lo_vec is not None and rhs.theta_hi_vec is not None:
            lo = torch.tensor(rhs.theta_lo_vec, dtype=torch.float32)
            hi = torch.tensor(rhs.theta_hi_vec, dtype=torch.float32)
        else:
            lo = torch.full((self.theta_dim,), theta_lo)
            hi = torch.full((self.theta_dim,), theta_hi)
        self.register_buffer("theta_lo_vec", lo)
        self.register_buffer("theta_hi_vec", hi)

        # `gru_y_cols`: indices of y to feed into the GRU. None = all P (legacy).
        # When restricted (e.g. to obs only), the GRU avoids being dominated by
        # unobservable latent states whose values are model fabrications.
        self.gru_y_cols = list(gru_y_cols) if gru_y_cols is not None else None
        self._has_gru_y_cols: bool = gru_y_cols is not None
        self.register_buffer(
            "gru_y_idx",
            torch.tensor(list(gru_y_cols) if gru_y_cols is not None else [], dtype=torch.long),
            persistent=False,
        )
        gru_y_dim = len(self.gru_y_cols) if self.gru_y_cols is not None else self.P

        gru_feat_dim = (len(self.gru_u_cols) if self.gru_u_cols is not None else self.U) * self._u_feat_mult
        feat_in = gru_feat_dim + gru_y_dim
        if self.encoder_use_time:
            feat_in += 1
        if self.encoder_use_log_dt:
            feat_in += 1
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
        head_out = self.theta_dim + self.P if self.use_basal else self.theta_dim
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
        self.head = nn.Linear(head_in, head_out)

        # Head init: "supervisor" applies xavier_uniform + zeros UNCONDITIONALLY.
        # "default" preserves the legacy guard (only override on non-default config values).
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

        # GRU init: "supervisor" forces orthogonal_(W_hh) + xavier_uniform_(W_ih) + zeros(biases)
        # on whichever variant is in use. nn.GRU stores params under names like
        # "weight_ih_l{k}", "weight_hh_l{k}", "bias_ih_l{k}", "bias_hh_l{k}".
        # StackedGRUCellBlock already applies this init at construction; redo to honor seed ordering.
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

        # y0 MLP: encodes the initial observation y0 into a per-sample logit bias added
        # to the GRU head output at every timestep.  Zero-initialized output layer so the
        # model starts as a standard GRU (no regression to the population mean at init).
        # Input uses the same y-column subset the GRU sees (gru_y_cols if set, else all P).
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

        if u_to_y_jump.shape != (self.U, self.P):
            raise ValueError(f"u_to_y_jump must be (U,P)=({self.U},{self.P}), got {tuple(u_to_y_jump.shape)}")
        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

        # Per-channel MinMax max (used by "minmax" / "minmax_sqrt" u_transform).
        # Built into a (U,) vector with 1.0 in non-scaled columns (e.g. DNA c)
        # so applying scale = 1/u_max_full leaves them unchanged.
        # Always register the buffer so TorchScript can resolve the attribute.
        # When unused, it stays at all-ones (a no-op divisor).
        u_max_full = torch.ones(self.U, dtype=torch.float32)
        if u_minmax_max is not None and u_minmax_cols is not None:
            cols = torch.as_tensor(u_minmax_cols, dtype=torch.long)
            u_max_full[cols] = u_minmax_max.float().clamp_min(1e-8)
            self._has_u_minmax = True
        else:
            self._has_u_minmax = False
        self.register_buffer("u_minmax_max_full", u_max_full, persistent=False)

        # ── idea #1: stage-evaluated neural residual g(y) ────────────────────
        # A small MLP P->P added to the RHS at every RK4 stage (so the residual
        # tracks y(t) inside the step, unlike the step-constant basal beta).
        # Last layer zero-init → starts as pure mechanism (matches the
        # NeuralOdeCorrection baseline). Always built (cheap) so TorchScript can
        # resolve the attribute; only USED when self.rk4_residual is True.
        self.rk4_residual = bool(rk4_residual)
        res_layers: list[nn.Module] = [nn.Linear(self.P, int(rk4_residual_hidden)), nn.SiLU()]
        for _ in range(int(rk4_residual_layers) - 1):
            res_layers += [nn.Linear(int(rk4_residual_hidden), int(rk4_residual_hidden)), nn.SiLU()]
        res_layers.append(nn.Linear(int(rk4_residual_hidden), self.P))
        nn.init.zeros_(res_layers[-1].weight)
        nn.init.zeros_(res_layers[-1].bias)
        self.rk4_residual_mlp = nn.Sequential(*res_layers)

    def _decay_trace(self, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """dt-aware multi-timescale leaky integrators.

        For each time constant τ, integrate  tr_k = exp(-dt_k/τ)·tr_{k-1} + u_k.
        The value at any step encodes how much was dosed AND how recently (it
        decays over τ). Stacking several τ's gives a fast→persistent view: the
        fast trace tracks event timing/recency, the slow trace approximates the
        cumulative recipe. Sequential scan (not the closed form) for numerical
        stability — exp(T/τ) would overflow for T~1e4 s, τ~1e2 s.

        u: (B,K,C), dt: (B,K)  →  (B,K, C*n_tau).  sqrt-compressed for scale.
        """
        B = u.shape[0]; K = u.shape[1]; C = u.shape[2]
        n_tau = self.u_decay_taus.shape[0]
        out = torch.zeros(B, K, C * n_tau, device=u.device, dtype=u.dtype)
        for ti in range(n_tau):
            tau = self.u_decay_taus[ti]
            decay = torch.exp(-dt / tau)                  # (B,K)
            tr = torch.zeros(B, C, device=u.device, dtype=u.dtype)
            for k in range(K):
                tr = decay[:, k].unsqueeze(1) * tr + u[:, k, :]
                out[:, k, ti * C:(ti + 1) * C] = tr
        return out.clamp_min(1e-6).sqrt()

    def _time_since(self, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Per-channel elapsed time since the last nonzero dose in that channel.

        Resets to 0 at each event step, grows by dt otherwise → "how long ago did
        this reagent last arrive." log1p-compressed (times reach ~1e4 s). Paired
        with cumsum (magnitude), gives magnitude + recency; the u_open column's
        trace is the time-since-tube-opening signal the new-data dynamics need.

        u: (B,K,C), dt: (B,K)  →  (B,K,C).
        """
        B = u.shape[0]; K = u.shape[1]; C = u.shape[2]
        out = torch.zeros(B, K, C, device=u.device, dtype=u.dtype)
        tsince = torch.zeros(B, C, device=u.device, dtype=u.dtype)
        event = u.abs() > 1e-8                            # (B,K,C)
        for k in range(K):
            tsince = tsince + dt[:, k].unsqueeze(1)
            tsince = torch.where(event[:, k, :], torch.zeros_like(tsince), tsince)
            out[:, k, :] = tsince
        return torch.log1p(out)

    def forward(
        self,
        y0: torch.Tensor,                     # (B,P)
        u_seq: torch.Tensor,                  # (B,K,U)
        dt_seq: torch.Tensor,                 # (B,K)
        obs_idx: torch.Tensor,                # (num_obs,) — pass torch.arange(P) for full TF
        y_seq: Optional[torch.Tensor] = None, # (B,K,P) for teacher forcing
        teacher_forcing: bool = True,
        tf_every: int = 50,
        u_transform: str = "none",            # GRU input transform:
                                              #   "none" | "cumsum" | "sqrt" | "cumsum_sqrt"
                                              #   "minmax" | "minmax_sqrt"  (require u_minmax_max at init)
        y_transform: str = "none",            # y_in transform: "none" | "sqrt" | "log1p"
                                              # Without a transform, large protein values (~10^3) dominate
                                              # the lift layer over reagent values (~1) and the GRU stops
                                              # responding to inputs. sqrt or log1p compresses the scale.
        theta_override: Optional[torch.Tensor] = None,  # (B,K,theta_dim) — if given, REPLACES the encoder's
                                              # emitted theta_k before integration (for parameter-gating
                                              # ablations: zero/floor, freeze-to-mean, scale). Encoder still
                                              # runs; only the value fed to the ODE is overridden.
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        y_out    = torch.empty(B, K, self.P, device=y0.device, dtype=y0.dtype)
        th_out   = torch.empty(B, K, self.theta_dim_emit, device=y0.device, dtype=y0.dtype)
        beta_out = torch.zeros(B, K, self.P, device=y0.device, dtype=y0.dtype)

        # Analytic scaffolds (e.g. IVTT) compute per-batch context (e.g. cumulative
        # DNA) once, and re-seed y_prev to a hidden-state-aware initial vector.
        # Typed as `Dict[str, Tensor]` (not `Optional[Dict]`) so TorchScript can
        # pass it into `analytic_step` without a None-narrow.
        analytic_ctx: Dict[str, torch.Tensor] = {}
        if self._analytic_scaffold:
            analytic_ctx = self.rhs.precompute_batch(y0, u_seq)

        h = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size, device=y0.device, dtype=y0.dtype)

        use_partial = obs_idx.numel() > 0

        # Pre-compute the GRU's view of u_seq (separate from the raw delta used for ODE jumps).
        # cumsum: after a bolus at step t, the GRU sees it at ALL subsequent steps — no long-range
        # memory required. The ODE jump always uses the raw delta u_seq.
        # NOTE: the model uses `self.u_transform` (fixed at init, sizes the lift layer), not the
        # forward arg — keeps the architecture and the feature build in lock-step.
        # Replaced `in (...)` membership tests with explicit `==` chains —
        # TorchScript handles plain string equality but not always the tuple membership form.
        if self._u_preselected:
            # Channel-expanding modes: select gru_u_cols ONCE (loop-invariant), then expand so the
            # feature carries persistent magnitude AND timing/recency. Loop skips re-selection.
            if self._has_gru_u_cols:
                u_base = torch.index_select(u_seq, 2, self.gru_u_idx)
            else:
                u_base = u_seq
            if self.u_transform == "pulse_cumsum_sqrt":
                pulse = u_base.clamp_min(1e-6).sqrt()                       # exact event timing+magnitude
                cum   = u_base.cumsum(dim=1).clamp_min(1e-6).sqrt()         # persistent running recipe
                u_gru = torch.cat([pulse, cum], dim=2)
            elif self.u_transform == "decay_trace":
                u_gru = self._decay_trace(u_base, dt_seq)                   # multi-timescale (B,K,C*3)
            elif self.u_transform == "pulse_cumsum_timesince":
                # Union of the two half-frontier winners: magnitude (old's driver) +
                # persistence + recency (new's driver). 3 channels per col.
                pulse  = u_base.clamp_min(1e-6).sqrt()                      # exact magnitude (old needs this)
                cum    = u_base.cumsum(dim=1).clamp_min(1e-6).sqrt()        # persistent recipe
                tsince = self._time_since(u_base, dt_seq)                   # recency (new benefits)
                u_gru  = torch.cat([pulse, cum, tsince], dim=2)
            elif self.u_transform == "pulse_cumsum_static":
                # pulse + progressive cumsum + STATIC full recipe (final total broadcast to all
                # steps). The recipe is a known control input, so giving the COMPLETE recipe from
                # t=0 (not just progressively) lets the encoder set theta correctly upfront. 3 ch.
                cum_raw = u_base.cumsum(dim=1)
                pulse   = u_base.clamp_min(1e-6).sqrt()
                cum     = cum_raw.clamp_min(1e-6).sqrt()
                total   = cum_raw[:, -1:, :].expand(-1, cum_raw.shape[1], -1).clamp_min(1e-6).sqrt()
                u_gru   = torch.cat([pulse, cum, total], dim=2)
            else:  # cumsum_timesince_sqrt
                cum    = u_base.cumsum(dim=1).clamp_min(1e-6).sqrt()        # magnitude
                tsince = self._time_since(u_base, dt_seq)                   # recency
                u_gru  = torch.cat([cum, tsince], dim=2)
        else:
            if self.u_transform == "cumsum" or self.u_transform == "cumsum_sqrt":
                u_gru = u_seq.cumsum(dim=1)
            else:
                u_gru = u_seq
            if self.u_transform == "minmax" or self.u_transform == "minmax_sqrt":
                if not self._has_u_minmax:
                    raise ValueError(
                        "u_transform=" + str(self.u_transform) + " requires u_minmax_max/u_minmax_cols at model init."
                    )
                u_gru = u_gru / self.u_minmax_max_full.view(1, 1, -1)
            if self.u_transform == "sqrt" or self.u_transform == "cumsum_sqrt" or self.u_transform == "minmax_sqrt":
                u_gru = u_gru.clamp_min(1e-6).sqrt()

        # Per-sample logit bias from y0 MLP (None when y0_theta_init=False).
        if self.y0_mlp is not None:
            y0_feat = torch.index_select(y0, dim=1, index=self.gru_y_idx) if self._has_gru_y_cols else y0
            if y_transform == "sqrt":
                y0_feat = y0_feat.clamp_min(0.0).sqrt()
            elif y_transform == "sqrt_clamp1":
                y0_feat = y0_feat.clamp_min(0.0).sqrt().clamp_min(1.0)
            elif y_transform == "log1p":
                y0_feat = torch.log1p(y0_feat.clamp_min(0.0))
            raw_y0_bias: Optional[torch.Tensor] = self.y0_mlp(y0_feat)  # (B, theta_dim)
        else:
            raw_y0_bias = None

        y_prev = self.rhs.initial_state(y0) if self._analytic_scaffold else y0
        for k in range(K):
            u_k     = u_seq[:, k, :]   # raw delta — used only for ODE jumps
            u_gru_k = u_gru[:, k, :]   # transformed — used for GRU features
            dt_k = dt_seq[:, k]

            y_in = y_prev.detach() if self.detach_y_prev else y_prev

            # Bob's verbatim policy: TF fires at k=0 too (with k-1=-1 wrapping to the
            # last frame). Default OdeRNN behaviour skips k=0.
            tf_fires = (k % tf_every == 0) if self._tf_at_k_zero else (k > 0 and k % tf_every == 0)
            if teacher_forcing and tf_fires and y_seq is not None:
                if use_partial:
                    y_in = y_prev.clone()
                    idx = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].to(dtype=y_in.dtype).detach()
                else:
                    y_in = y_seq[:, k - 1, :].to(dtype=y_prev.dtype).detach()

            # Use index_select on the precomputed long buffer instead of advanced
            # indexing with a Python list (TorchScript can't reliably script `t[:, list]`).
            # Pre-selected expanding modes already applied gru_u_cols in the pre-loop.
            if self._u_preselected:
                u_gru_k_feat = u_gru_k
            else:
                u_gru_k_feat = torch.index_select(u_gru_k, dim=1, index=self.gru_u_idx) if self._has_gru_u_cols else u_gru_k

            y_in_feat = torch.index_select(y_in, dim=1, index=self.gru_y_idx) if self._has_gru_y_cols else y_in
            if y_transform == "sqrt":
                y_in_feat = y_in_feat.clamp_min(1e-6).sqrt()
            elif y_transform == "sqrt_clamp1":
                # Clamp to 1.0 BEFORE sqrt so gradient = 1/(2*sqrt(x)) stays finite.
                # sqrt(0) has gradient 1/(2*0)=inf which NaN-s the backward pass.
                # clamp_min(1.0) preserves the forward output (values <1 → 1.0, same
                # as the old clamp_min(0).sqrt().clamp_min(1.0) path) while making
                # the gradient 0 for unexpressed states (x<1, gradient of clamp=0).
                y_in_feat = y_in_feat.clamp_min(1.0).sqrt()
            elif y_transform == "log1p":
                y_in_feat = torch.log1p(y_in_feat.clamp_min(0.0))
            feat_parts = [u_gru_k_feat, y_in_feat]
            if self.encoder_use_time:
                tau_k = torch.full(
                    (u_gru_k_feat.shape[0], 1),
                    float(k) / float(max(K - 1, 1)),
                    device=u_gru_k_feat.device,
                    dtype=u_gru_k_feat.dtype,
                )
                feat_parts.append(tau_k)
            if self.encoder_use_log_dt:
                log_dt_k = torch.log(dt_k.clamp_min(1e-6)).to(dtype=u_gru_k_feat.dtype).unsqueeze(-1)
                feat_parts.append(log_dt_k)
            feat = torch.cat(feat_parts, dim=-1)
            x = self.lift(feat).unsqueeze(1)
            z, h = self.gru(x, h)
            raw = self.head(self.head_bottle(z.squeeze(1)))
            if raw_y0_bias is not None:
                raw = raw + raw_y0_bias if not self.use_basal else torch.cat(
                    [raw[:, :self.theta_dim] + raw_y0_bias, raw[:, self.theta_dim:]], dim=-1
                )

            if self.use_basal:
                raw_theta = raw[:, :self.theta_dim]
                if self.theta_bounded:
                    if self.theta_head_transform == "gamma":
                        theta_k = gamma(raw_theta, self.theta_lo_vec, self.theta_hi_vec)
                    else:
                        theta_k = log_gamma(raw_theta, self.theta_lo_vec, self.theta_hi_vec, tau=self.theta_head_tau)
                else:
                    theta_k = F.softplus(raw_theta)
                if theta_override is not None:
                    theta_k = theta_override[:, k, :]
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
                if theta_override is not None:
                    theta_k = theta_override[:, k, :]
                if self._analytic_scaffold:
                    # Closed-form integrator owns the step (no u-jump, no RK4).
                    y = self.rhs.analytic_step(y_prev, dt_k, theta_k, analytic_ctx)
                else:
                    y = y_prev + (u_k @ self.u_to_y_jump)
                    if self.rk4_residual:
                        y = self._rk4_substeps_residual(y, dt_k, theta_k)
                    else:
                        y = self._rk4_substeps(y, dt_k, theta_k)

            y_out[:, k, :] = y
            # When the scaffold defines a wider loss-facing theta layout (e.g. Bob's
            # 8-d [VTX, kdm, VTL, kmt, kmatm, R, lam, lamO]), let it repack here.
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
            y  = y.clamp(0.0, 1e5)
        return y

    def _rk4_substeps_residual(
        self,
        y: torch.Tensor,
        dt: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        # Idea #1: RHS = mechanism(theta) + g(y), with g re-evaluated at each RK4
        # stage so the neural residual tracks the integrated state within the step
        # (the UDE form), rather than being a step-constant add like basal beta.
        rhs = self.rhs
        g = self.rk4_residual_mlp
        n_sub = self.n_substeps
        dt = dt.unsqueeze(1)
        hdt = dt / float(n_sub)
        for _ in range(n_sub):
            k1 = rhs(y,                   theta) + g(y)
            k2 = rhs(y + 0.5 * hdt * k1, theta) + g(y + 0.5 * hdt * k1)
            k3 = rhs(y + 0.5 * hdt * k2, theta) + g(y + 0.5 * hdt * k2)
            k4 = rhs(y +       hdt * k3,  theta) + g(y +       hdt * k3)
            y  = y + (hdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            y  = y.clamp(0.0, 1e5)
        return y

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
