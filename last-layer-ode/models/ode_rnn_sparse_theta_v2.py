"""Single-pass sparse-θ model.

Implements Experiment B from new_scaffolds.tex faithfully: the encoder is
constrained to emit a kinetic vector at only K anchor positions over the
experiment, not at every timestep. Between anchors θ(t) is held piecewise
constant.

Differs from `ode_rnn_sparse_theta` (the two-pass wrapper):
  - One forward pass instead of two (≈ 2× faster).
  - The K-vector constraint is architectural — the θ-head literally fires only
    K times per experiment, instead of firing 200 times and being subsampled
    post hoc. That's the constraint the tex describes ("forces the model to
    output only K kinetic vectors for the full experiment").
  - JIT-scriptable: same loop structure as OdeRNN, just adds a per-step
    "is this an anchor?" gate that the bounded-θ recomputation reads.

Trade-off:
  - Only piecewise interpolation supported (linear-interp anchor scheduling
    needs a forward look-ahead and isn't single-pass; for B3 keep using
    OdeRNNSparseTheta with anchor_interp="linear").

Subclasses OdeRNN: inherits __init__, GRU/head construction, JIT-friendly
attribute layout, encoder feature pipeline, teacher-forcing logic, and the
analytic/RK4 integration step. Only `forward()` is overridden.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from models.ode_rnn import OdeRNN, gamma, log_gamma
import torch.nn.functional as F


def _anchor_mask(K: int, n_anchors: int, device: torch.device) -> torch.Tensor:
    """Boolean (K,) mask marking the K anchor timesteps.

    Same equally-spaced policy as the two-pass wrapper: positions
    round(linspace(0, K-1, n_anchors)). Step 0 is always an anchor.
    """
    if int(n_anchors) < 1:
        raise ValueError(f"n_theta_anchors must be >= 1, got {n_anchors}")
    mask = torch.zeros(K, dtype=torch.bool, device=device)
    if int(n_anchors) == 1:
        mask[0] = True
        return mask
    pos = torch.linspace(0, K - 1, steps=int(n_anchors), dtype=torch.float32,
                         device=device).round().long()
    mask[pos] = True
    return mask


def _bolus_anchor_mask(u_seq: torch.Tensor, max_anchors: int) -> torch.Tensor:
    """Per-sample anchor mask (B, K) bool, anchors at bolus events.

    A bolus event = any timestep where the raw input vector has a nonzero entry.
    k=0 is always marked as an anchor (need a θ to start from). If `max_anchors`
    is > 0, keep only the first `max_anchors` True positions per sample (the
    rest of the bolus events still feed the GRU encoder but don't trigger a new
    θ anchor — θ holds the last anchor's value through them).
    """
    B, K, _ = u_seq.shape
    bolus = (u_seq.abs().sum(dim=-1) > 0)  # (B, K)
    mask = bolus.clone()
    mask[:, 0] = True
    if int(max_anchors) > 0:
        cnt = mask.long().cumsum(dim=1)
        mask = mask & (cnt <= int(max_anchors))
    return mask


class OdeRNNSparseThetaV2(OdeRNN):
    """Single-pass sparse-θ on top of OdeRNN (piecewise interp only)."""

    __constants__ = OdeRNN.__constants__

    def __init__(self, *, n_theta_anchors: int = 6,
                 anchor_mode: str = "uniform",
                 bolus_max_anchors: int = 0,
                 **kwargs):
        # OdeRNN already supports encoder_use_time, basal, lift_skip, etc. — all
        # forwarded via kwargs untouched.
        OdeRNN.__init__(self, **kwargs)
        if int(n_theta_anchors) < 1:
            raise ValueError(f"n_theta_anchors must be >= 1, got {n_theta_anchors}")
        if anchor_mode not in ("uniform", "bolus"):
            raise ValueError(f"anchor_mode must be 'uniform' or 'bolus', got {anchor_mode!r}")
        self.n_theta_anchors = int(n_theta_anchors)
        # "uniform" = linspace(0, K-1, n_theta_anchors) anchors (the tex default).
        # "bolus"   = per-sample anchors at bolus events from u_seq; n_theta_anchors
        #             is ignored in this mode (use bolus_max_anchors to cap).
        self.anchor_mode = str(anchor_mode)
        # Cap on bolus anchors per sample. 0 = no cap (every bolus is an anchor).
        self.bolus_max_anchors = int(bolus_max_anchors)

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
        y_out    = torch.empty(B, K, self.P, device=y0.device, dtype=y0.dtype)
        th_out   = torch.empty(B, K, self.theta_dim_emit, device=y0.device, dtype=y0.dtype)
        beta_out = torch.zeros(B, K, self.P, device=y0.device, dtype=y0.dtype)

        analytic_ctx: Dict[str, torch.Tensor] = {}
        if self._analytic_scaffold:
            analytic_ctx = self.rhs.precompute_batch(y0, u_seq)

        h = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size, device=y0.device, dtype=y0.dtype)
        use_partial = obs_idx.numel() > 0

        # Encoder u features (cumsum / sqrt / minmax pipeline) — same as OdeRNN.
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

        # y0-MLP bias (same as OdeRNN).
        if self.y0_mlp is not None:
            y0_feat = torch.index_select(y0, dim=1, index=self.gru_y_idx) if self._has_gru_y_cols else y0
            if y_transform == "sqrt":
                y0_feat = y0_feat.clamp_min(0.0).sqrt()
            elif y_transform == "sqrt_clamp1":
                y0_feat = y0_feat.clamp_min(0.0).sqrt().clamp_min(1.0)
            elif y_transform == "log1p":
                y0_feat = torch.log1p(y0_feat.clamp_min(0.0))
            raw_y0_bias: Optional[torch.Tensor] = self.y0_mlp(y0_feat)
        else:
            raw_y0_bias = None

        # Precompute the anchor positions for this batch.
        #   uniform: (K,) bool, same for every sample (linspace anchors).
        #   bolus:   (B,K) bool, per-sample — anchor at each bolus event in u_seq
        #            (plus k=0). In bolus mode the θ-head fires every step and
        #            theta_cur is updated only for samples whose mask fires,
        #            via torch.where (no Python-scalar branch on per-sample state).
        if self.anchor_mode == "bolus":
            anchor_mask_bs = _bolus_anchor_mask(u_seq, self.bolus_max_anchors)  # (B,K)
            is_anchor = torch.zeros(K, dtype=torch.bool, device=y0.device)       # unused in bolus
        else:
            is_anchor = _anchor_mask(K, self.n_theta_anchors, y0.device)         # (K,)
            anchor_mask_bs = torch.zeros(B, K, dtype=torch.bool, device=y0.device)  # unused

        # `theta_cur` and `beta_cur` carry the last anchor's value forward.
        # Allocated lazily at k=0 (first anchor) so dtype/device match the head output.
        theta_cur: Optional[torch.Tensor] = None
        beta_cur: Optional[torch.Tensor] = None

        y_prev = self.rhs.initial_state(y0) if self._analytic_scaffold else y0
        for k in range(K):
            u_k     = u_seq[:, k, :]
            u_gru_k = u_gru[:, k, :]
            dt_k = dt_seq[:, k]

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
            if y_transform == "sqrt":
                y_in_feat = y_in_feat.clamp_min(0.0).sqrt()
            elif y_transform == "sqrt_clamp1":
                y_in_feat = y_in_feat.clamp_min(0.0).sqrt().clamp_min(1.0)
            elif y_transform == "log1p":
                y_in_feat = torch.log1p(y_in_feat.clamp_min(0.0))
            feat_parts = [u_gru_k_feat, y_in_feat]
            if self.encoder_use_log_dt:
                log_dt_k = torch.log(dt_k.clamp_min(1e-6)).to(dtype=u_gru_k_feat.dtype).unsqueeze(-1)
                feat_parts.append(log_dt_k)
            if self.encoder_use_time:
                tau_k = torch.full(
                    (u_gru_k_feat.shape[0], 1),
                    float(k) / float(max(K - 1, 1)),
                    device=u_gru_k_feat.device,
                    dtype=u_gru_k_feat.dtype,
                )
                feat_parts.append(tau_k)
            feat = torch.cat(feat_parts, dim=-1)
            x = self.lift(feat).unsqueeze(1)
            z, h = self.gru(x, h)

            # === Sparse-θ: fire the θ-head only at anchor steps. ===
            # `theta_cur` and `beta_cur` from a prior anchor are reused on non-anchor
            # steps (piecewise-constant θ(t) between anchors). The GRU still steps
            # every k so its hidden state carries the full input/output history.
            #
            # uniform mode: scalar Python branch on is_anchor[k] (fast — head fires
            #               only K times across the trajectory).
            # bolus mode:   per-sample anchor mask, so the head must fire every step
            #               and theta_cur is updated for the subset of samples whose
            #               mask fires this step via torch.where. Slower but per-sample.
            if self.anchor_mode == "bolus":
                raw = self.head(self.head_bottle(z.squeeze(1)))
                if raw_y0_bias is not None:
                    raw = raw + raw_y0_bias if not self.use_basal else torch.cat(
                        [raw[:, :self.theta_dim] + raw_y0_bias, raw[:, self.theta_dim:]], dim=-1
                    )
                if self.use_basal:
                    raw_theta = raw[:, :self.theta_dim]
                    if self.theta_bounded:
                        if self.theta_head_transform == "gamma":
                            theta_new = gamma(raw_theta, self.theta_lo_vec, self.theta_hi_vec)
                        else:
                            theta_new = log_gamma(raw_theta, self.theta_lo_vec, self.theta_hi_vec, tau=self.theta_head_tau)
                    else:
                        theta_new = F.softplus(raw_theta)
                    beta_new = raw[:, self.theta_dim:] * (y_prev / (y_prev + 1.0))
                else:
                    if self.theta_bounded:
                        if self.theta_head_transform == "gamma":
                            theta_new = gamma(raw, self.theta_lo_vec, self.theta_hi_vec)
                        else:
                            theta_new = log_gamma(raw, self.theta_lo_vec, self.theta_hi_vec, tau=self.theta_head_tau)
                    else:
                        theta_new = F.softplus(raw)
                    beta_new = torch.zeros(0, device=raw.device, dtype=raw.dtype)  # unused

                fires_b = anchor_mask_bs[:, k].unsqueeze(-1)  # (B,1) bool
                if theta_cur is None:
                    # First step: nothing to hold over, take theta_new everywhere.
                    theta_cur = theta_new
                    if self.use_basal:
                        beta_cur = beta_new
                else:
                    theta_cur = torch.where(fires_b, theta_new, theta_cur)
                    if self.use_basal:
                        assert beta_cur is not None
                        beta_cur = torch.where(fires_b, beta_new, beta_cur)
            else:
                if bool(is_anchor[k].item()) or theta_cur is None:
                    raw = self.head(self.head_bottle(z.squeeze(1)))
                    if raw_y0_bias is not None:
                        raw = raw + raw_y0_bias if not self.use_basal else torch.cat(
                            [raw[:, :self.theta_dim] + raw_y0_bias, raw[:, self.theta_dim:]], dim=-1
                        )
                    if self.use_basal:
                        raw_theta = raw[:, :self.theta_dim]
                        if self.theta_bounded:
                            if self.theta_head_transform == "gamma":
                                theta_cur = gamma(raw_theta, self.theta_lo_vec, self.theta_hi_vec)
                            else:
                                theta_cur = log_gamma(raw_theta, self.theta_lo_vec, self.theta_hi_vec, tau=self.theta_head_tau)
                        else:
                            theta_cur = F.softplus(raw_theta)
                        beta_cur = raw[:, self.theta_dim:] * (y_prev / (y_prev + 1.0))
                    else:
                        if self.theta_bounded:
                            if self.theta_head_transform == "gamma":
                                theta_cur = gamma(raw, self.theta_lo_vec, self.theta_hi_vec)
                            else:
                                theta_cur = log_gamma(raw, self.theta_lo_vec, self.theta_hi_vec, tau=self.theta_head_tau)
                        else:
                            theta_cur = F.softplus(raw)

            theta_k = theta_cur  # piecewise: hold last anchor's value

            if self.use_basal:
                assert beta_cur is not None
                beta_k = beta_cur
                beta_out[:, k, :] = beta_k
                y = y_prev + (u_k @ self.u_to_y_jump)
                y = self._rk4_substeps_basal(y, dt_k, theta_k, beta_k)
            else:
                if self._analytic_scaffold:
                    y = self.rhs.analytic_step(y_prev, dt_k, theta_k, analytic_ctx)
                else:
                    y = y_prev + (u_k @ self.u_to_y_jump)
                    y = self._rk4_substeps(y, dt_k, theta_k)

            y_out[:, k, :] = y
            th_out[:, k, :] = self.rhs.emit_theta(theta_k, y) if self._analytic_scaffold else theta_k
            y_prev = y

        return y_out, th_out, beta_out
