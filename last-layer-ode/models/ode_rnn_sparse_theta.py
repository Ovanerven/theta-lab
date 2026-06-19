"""K-anchor sparse-θ wrapper for OdeRNN (the K-anchor CMVF).

Implements the sparse-readout variant from new_scaffolds.tex §3.4 (Experiment B):
instead of one kinetic vector per timestep, the encoder is sampled at
`n_theta_anchors` equally-spaced anchor timesteps and θ(t) is reconstructed
between anchors by piecewise-constant or linear interpolation, then the ODE
is re-integrated with that anchored θ(t).

Registered as `ode_rnn_sparse_theta` (on top of the regular OdeRNN).

It accepts the same two extra kwargs:
    n_theta_anchors: int  (>=1; 1 = single global θ, 6 = stage-like, etc.)
    anchor_interp:   "piecewise" | "linear"

Design — two-pass per forward:
    Pass 1: run the underlying encoder's forward() to extract its dense θ(t).
            The integrated y_pred from this pass is discarded.
    Pass 2: sample θ at K anchor positions, reconstruct θ(t), and re-integrate
            the scaffold using each base class's own RK4 / analytic-step path.

Compute cost is roughly 2× a normal forward, which is acceptable for the
baseline-comparison runs this targets.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from models.ode_rnn import OdeRNN


# ============================================================================
# Anchor helpers (architecture-independent)
# ============================================================================

def _anchor_indices(n_theta_anchors: int, K: int, device: torch.device) -> torch.Tensor:
    """K equally-spaced anchor positions in [0, K-1], including both endpoints."""
    if int(n_theta_anchors) < 1:
        raise ValueError(f"n_theta_anchors must be >= 1, got {n_theta_anchors}")
    if int(n_theta_anchors) == 1:
        return torch.tensor([0], dtype=torch.long, device=device)
    return torch.linspace(
        0, K - 1, steps=int(n_theta_anchors), dtype=torch.float32, device=device
    ).round().long()


def _expand_anchors(
    anchor_theta: torch.Tensor,  # (B, n_anchors, theta_dim)
    K: int,
    anchor_interp: str,
) -> torch.Tensor:
    """Build (B, K, theta_dim) from K anchor vectors via piecewise or linear interp."""
    if anchor_interp not in {"piecewise", "linear"}:
        raise ValueError(f"anchor_interp must be 'piecewise' or 'linear', got {anchor_interp!r}")

    B, n, D = anchor_theta.shape
    device = anchor_theta.device
    anchor_pos = _anchor_indices(n, K, device).to(dtype=torch.float32)  # (n,)
    ks = torch.arange(K, device=device, dtype=torch.float32)            # (K,)

    if anchor_interp == "piecewise":
        le = (ks.unsqueeze(1) >= anchor_pos.unsqueeze(0))               # (K, n)
        left_idx = le.float().sum(dim=1).clamp(min=1).long() - 1        # (K,)
        return anchor_theta[:, left_idx, :]                             # (B, K, D)

    # linear
    le = (ks.unsqueeze(1) >= anchor_pos.unsqueeze(0))
    left_idx = (le.float().sum(dim=1).clamp(min=1).long() - 1).clamp(max=n - 2)
    right_idx = left_idx + 1
    left_pos = anchor_pos[left_idx]
    right_pos = anchor_pos[right_idx]
    span = (right_pos - left_pos).clamp(min=1.0)
    alpha = ((ks - left_pos) / span).clamp(0.0, 1.0)                    # (K,)
    left_theta = anchor_theta[:, left_idx, :]                           # (B, K, D)
    right_theta = anchor_theta[:, right_idx, :]                         # (B, K, D)
    return left_theta + alpha.view(1, K, 1) * (right_theta - left_theta)


# ============================================================================
# Shared mixin: pass-2 re-integration loop.
#
# Each concrete wrapper picks `_step` (analytic vs. RK4) based on the scaffold.
# The non-basal RK4 path uses `_rk4_substeps(y, dt, theta)`; the basal class
# only has `_rk4_substeps_basal(y, dt, theta, beta)` so we pass beta=0 for it.
# ============================================================================

class _SparseThetaMixin:
    """Provides the two-pass anchored forward; combine with a concrete encoder base."""

    n_theta_anchors: int
    anchor_interp: str
    _use_basal_rk4: bool  # True for OdeRNNBasalV2-based wrapper

    def _sparse_init(self, n_theta_anchors: int, anchor_interp: str, use_basal_rk4: bool):
        if int(n_theta_anchors) < 1:
            raise ValueError(f"n_theta_anchors must be >= 1, got {n_theta_anchors}")
        if anchor_interp not in {"piecewise", "linear"}:
            raise ValueError(f"anchor_interp must be 'piecewise' or 'linear', got {anchor_interp!r}")
        self.n_theta_anchors = int(n_theta_anchors)
        self.anchor_interp = str(anchor_interp)
        self._use_basal_rk4 = bool(use_basal_rk4)

    def _reintegrate_with_anchored_theta(
        self,
        y0: torch.Tensor,
        u_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        theta_eff: torch.Tensor,
    ) -> torch.Tensor:
        """Run pass 2: integrate scaffold using theta_eff (B,K,theta_dim)."""
        B, K, _ = theta_eff.shape
        y_out = torch.empty(B, K, self.P, device=y0.device, dtype=y0.dtype)
        th_out = torch.empty(B, K, self.theta_dim_emit, device=y0.device, dtype=y0.dtype)

        analytic_ctx: Dict[str, torch.Tensor] = {}
        if self._analytic_scaffold:
            analytic_ctx = self.rhs.precompute_batch(y0, u_seq)

        y_prev = self.rhs.initial_state(y0) if self._analytic_scaffold else y0

        for k in range(K):
            theta_k = theta_eff[:, k, :]
            u_k = u_seq[:, k, :]
            dt_k = dt_seq[:, k]

            if self._analytic_scaffold:
                y = self.rhs.analytic_step(y_prev, dt_k, theta_k, analytic_ctx)
                y = torch.clamp_min(y, 0.0)
            else:
                y = y_prev + (u_k @ self.u_to_y_jump)
                if self._use_basal_rk4:
                    zero_beta = torch.zeros_like(y)
                    y = self._rk4_substeps_basal(y, dt_k, theta_k, zero_beta)
                else:
                    y = self._rk4_substeps(y, dt_k, theta_k)

            y_out[:, k, :] = y
            th_out[:, k, :] = self.rhs.emit_theta(theta_k, y) if self._analytic_scaffold else theta_k
            y_prev = y

        beta_out = torch.zeros(B, K, self.P, device=y0.device, dtype=y0.dtype)
        return y_out, th_out, beta_out

    def _anchored_forward(
        self,
        super_forward,         # bound method of the underlying base class
        y0, u_seq, dt_seq, obs_idx, y_seq,
        **base_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pass 1: extract dense θ from the underlying encoder. y_pred from this
        # pass is discarded.
        _, theta_dense, _ = super_forward(
            y0=y0, u_seq=u_seq, dt_seq=dt_seq, obs_idx=obs_idx, y_seq=y_seq,
            **base_kwargs,
        )
        B, K, D = theta_dense.shape

        # Anchor + interp.
        anchor_ks = _anchor_indices(self.n_theta_anchors, K, theta_dense.device)
        anchor_theta = theta_dense.index_select(dim=1, index=anchor_ks)  # (B, n_anchors, D)
        theta_eff = _expand_anchors(anchor_theta, K, self.anchor_interp)

        # Pass 2: re-integrate.
        return self._reintegrate_with_anchored_theta(y0, u_seq, dt_seq, theta_eff)


# ============================================================================
# Concrete wrappers (one per working base architecture)
# ============================================================================

class OdeRNNSparseTheta(_SparseThetaMixin, OdeRNN):
    """K-anchor sparse-θ on top of the regular OdeRNN."""

    def __init__(self, *, n_theta_anchors: int = 6, anchor_interp: str = "piecewise", **kwargs):
        OdeRNN.__init__(self, **kwargs)
        self._sparse_init(n_theta_anchors, anchor_interp, use_basal_rk4=False)

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
    ):
        return self._anchored_forward(
            OdeRNN.forward.__get__(self, OdeRNN),
            y0, u_seq, dt_seq, obs_idx, y_seq,
            teacher_forcing=teacher_forcing, tf_every=tf_every,
            u_transform=u_transform, y_transform=y_transform,
        )

