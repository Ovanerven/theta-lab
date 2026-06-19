#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone CMVF GRU model, M5 scaffold (run HSa_M5_h400_s2).

A self-contained closed-loop hybrid neural-ODE: at each step a GRU emits a
kinetic vector theta_k, a bolus jump injects added reagents, and an RK4 step
integrates the M5 TXTL ODE over dt_k. The GRU gives time-varying rates; the
scaffold gives the mass-action structure. Observed channels: mRNA (mm), protein (pm).

M5 scaffold (states [R, O, m, mm, p, pm, DNA], theta [lam, lam_O, VTXmax, kdm, VTLmax, kmt, kmatm]):
    dR=-lam*R;  dO=-lam_O*O;  dm=R*VTXmax*DNA-(kdm+kmatm)*m;  dmm=kmatm*m-kdm*mm
    dp=R*VTLmax*(m+mm)-kmt*p;  dpm=O*kmt*p;  dDNA=0 (bolus-driven only)

Run:
    python train_cmvf_gru_M5.py              # full 200-epoch run
    python train_cmvf_gru_M5.py --epochs 5   # quick smoke test
    python train_cmvf_gru_M5.py --data /path/to/dataset.npz

Prints held-out test protein R^2 (final pm) and mRNA-max R^2 (peak mm), raw units.
Reference (200 epochs, seed 2): protein R^2 ~ 0.66, mRNA-max R^2 ~ 0.24.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 1. CONFIG: hyperparameters for run HSa_M5_h400_s2 (from the original YAML) ---
@dataclass
class Config:
    # --- data / split ---
    dataset_path: str = "datasets/cell-free/txtl_native_real_only_coarsenold.npz"
    val_n: int = 125
    test_n: int = 125
    split_seed: int = 42
    stratified_split: bool = True
    stratify_bins: int = 5
    stratify_targets: tuple = (3, 5)          # mm, pm endpoints
    stratify_by_source: bool = True           # split OLD/NEW pools independently

    # --- optimisation ---
    epochs: int = 200
    batch_size: int = 100
    lr: float = 0.002
    weight_decay: float = 0.0018
    warmup_epochs: int = 5
    cosine_decay: bool = False
    seed: int = 2
    grad_clip: float = 1.0

    # --- teacher forcing schedule ---
    teacher_forcing: bool = True
    tf_every: int = 200
    tf_drop_epoch: int = 100

    # --- architecture ---
    hidden: int = 400
    lift_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.2
    n_substeps: int = 3
    theta_lo: float = 1e-6                     # scalar fallback; scaffold vec wins
    theta_hi: float = 2.0
    theta_bounded: bool = True
    lift_skip: bool = True
    gru_init: str = "orthogonal"              # == "supervisor" in the old yaml
    head_init: str = "orthogonal"             # == "supervisor" in the old yaml
    encoder_use_time: bool = False
    encoder_use_log_dt: bool = True
    theta_head_transform: str = "log_gamma"
    theta_head_tau: float = 2.3

    # --- feature transforms ---
    u_transform: str = "pulse_cumsum_sqrt"
    y_transform: str = "sqrt_clamp1"

    # --- column selections (indices into the dataset arrays) ---
    obs_idx: tuple = (3, 5)                    # supervised channels: mm, pm
    gru_y_cols: tuple = (3, 5)                 # y channels fed to the GRU
    gru_u_cols: tuple = (0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11)  # u channels fed to the GRU

    # --- channel-min "gate the observation" ---
    subtract_channel_min: bool = True
    subtract_channel_min_cols: tuple = (3, 5)

    # --- loss ---
    source_loss_weights: dict = field(default_factory=lambda: {"new": 1.0, "old": 1.0})
    loss_normalizer_channels: int = 3
    loss_clamp_min: float = 1.0


CONFIG = Config()


# --- 2a. Mechanistic scaffold: forward(y, theta) -> dy/dt (fixed mass-action ODE) ---
class M5Scaffold(nn.Module):
    """M5 TXTL scaffold. States [R, O, m, mm, p, pm, DNA]; theta [lam, lam_O, VTXmax, kdm, VTLmax, kmt, kmatm]."""

    def __init__(self):
        super().__init__()
        self.P = 7
        self.theta_dim = 7
        self.state_names = ["R", "O", "m", "mm", "p", "pm", "DNA"]
        # log-uniform per-parameter bounds (theta is squashed into these).
        self.theta_lo_vec = [1e-6, 1e-6, 3e-5, 1e-5, 3e-5, 1e-5, 5e-5]
        self.theta_hi_vec = [5e-4, 5e-4, 1.2e-1, 1e-2, 8e-2, 3.5e-4, 3.5e-3]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        R, O, m, mm, p, pm, DNA = y.unbind(dim=-1)
        lam, lam_O, VTXmax, kdm, VTLmax, kmt, kmatm = theta.unbind(dim=-1)

        # Clamp states at 0 inside the RHS (avoid nonphysical negative-rate feedback).
        R_p = torch.clamp_min(R, 0.0)
        O_p = torch.clamp_min(O, 0.0)
        m_p = torch.clamp_min(m, 0.0)
        mm_p = torch.clamp_min(mm, 0.0)
        p_p = torch.clamp_min(p, 0.0)
        DNA_p = torch.clamp_min(DNA, 0.0)

        dR = -lam * R_p
        dO = -lam_O * O_p
        dm = R_p * VTXmax * DNA_p - (kdm + kmatm) * m_p
        dmm = kmatm * m_p - kdm * mm_p
        dp = R_p * VTLmax * (m_p + mm_p) - kmt * p_p
        dpm = O_p * kmt * p_p
        dDNA = torch.zeros_like(DNA)

        return torch.stack((dR, dO, dm, dmm, dp, dpm, dDNA), dim=-1)


# --- 2b. Theta head transforms: squash head logits into [lo, hi] ---
def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    # Sigmoid in log-space; tau>1 (=2.3) flattens it so the head can't saturate.
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x / tau))


def gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    # Linear-sigmoid (kept for completeness; M5 uses log_gamma).
    return lo + (hi - lo) * torch.sigmoid(x)


# --- 2c. GRU encoder + theta head + RK4 closed loop ---
class OdeRNN(nn.Module):
    def __init__(self, *, U: int, rhs: M5Scaffold, u_to_y_jump: torch.Tensor, cfg: Config):
        super().__init__()
        self.U = int(U)
        self.P = int(rhs.P)
        self.theta_dim = int(rhs.theta_dim)
        self.rhs = rhs
        self.cfg = cfg

        self.n_substeps = int(cfg.n_substeps)
        self.theta_lo = float(cfg.theta_lo)
        self.theta_hi = float(cfg.theta_hi)
        self.theta_bounded = bool(cfg.theta_bounded)
        self.theta_head_transform = str(cfg.theta_head_transform)
        self.theta_head_tau = float(cfg.theta_head_tau)
        self.lift_skip = bool(cfg.lift_skip)
        self.u_transform = str(cfg.u_transform)
        self.encoder_use_time = bool(cfg.encoder_use_time)
        self.encoder_use_log_dt = bool(cfg.encoder_use_log_dt)
        self.detach_y_prev = True

        # Per-parameter theta bounds from the scaffold (override the scalar).
        lo = torch.tensor(rhs.theta_lo_vec, dtype=torch.float32)
        hi = torch.tensor(rhs.theta_hi_vec, dtype=torch.float32)
        self.register_buffer("theta_lo_vec", lo)
        self.register_buffer("theta_hi_vec", hi)

        # Which u / y columns feed the GRU encoder.
        self.register_buffer("gru_u_idx", torch.tensor(list(cfg.gru_u_cols), dtype=torch.long))
        self.register_buffer("gru_y_idx", torch.tensor(list(cfg.gru_y_cols), dtype=torch.long))
        gru_y_dim = len(cfg.gru_y_cols)

        # u_transform = pulse_cumsum_sqrt doubles the u feature width (pulse + cumsum).
        u_feat_mult = 2 if self.u_transform == "pulse_cumsum_sqrt" else 1
        gru_feat_dim = len(cfg.gru_u_cols) * u_feat_mult

        feat_in = gru_feat_dim + gru_y_dim
        if self.encoder_use_time:
            feat_in += 1
        if self.encoder_use_log_dt:
            feat_in += 1

        # lift_skip: feed the feature straight into the GRU (no lift MLP).
        if self.lift_skip:
            self.lift = nn.Identity()
            gru_input_dim = feat_in
        else:
            self.lift = nn.Sequential(
                nn.Linear(feat_in, cfg.lift_dim),
                nn.SiLU(),
                nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
            )
            gru_input_dim = cfg.lift_dim

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=cfg.hidden,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(cfg.hidden, self.theta_dim)

        # --- weight init (gru_init / head_init = "orthogonal") ---
        nn.init.xavier_uniform_(self.head.weight, gain=1.0)
        nn.init.zeros_(self.head.bias)
        for name, p in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

        # Bolus jump matrix (U, P): adds reagent deltas into mechanistic states.
        assert u_to_y_jump.shape == (self.U, self.P)
        self.register_buffer("u_to_y_jump", u_to_y_jump.float())

    # RK4 integrator; theta is held constant across the n_substeps of one step.
    def _rk4_substeps(self, y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
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
            y = y.clamp(0.0, 1e5)
        return y

    def forward(
        self,
        y0: torch.Tensor,        # (B, P)
        u_seq: torch.Tensor,     # (B, K, U)
        dt_seq: torch.Tensor,    # (B, K)
        obs_idx: torch.Tensor,   # (num_obs,) supervised channels (used for TF)
        y_seq=None,              # (B, K, P) for teacher forcing
        teacher_forcing: bool = True,
        tf_every: int = 200,
        y_transform: str = "none",
    ):
        B, K, _ = u_seq.shape
        y_out = torch.empty(B, K, self.P, device=y0.device, dtype=y0.dtype)
        th_out = torch.empty(B, K, self.theta_dim, device=y0.device, dtype=y0.dtype)

        h = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size, device=y0.device, dtype=y0.dtype)
        use_partial = obs_idx.numel() > 0

        # pulse_cumsum_sqrt u-feature: [sqrt(pulse), sqrt(cumsum)] (raw u_seq is used for the jump).
        u_base = torch.index_select(u_seq, 2, self.gru_u_idx)
        pulse = u_base.clamp_min(1e-6).sqrt()
        cum = u_base.cumsum(dim=1).clamp_min(1e-6).sqrt()
        u_gru = torch.cat([pulse, cum], dim=2)

        y_prev = y0
        for k in range(K):
            u_k = u_seq[:, k, :]      # raw delta — ODE jump only
            u_gru_k = u_gru[:, k, :]  # transformed — GRU feature
            dt_k = dt_seq[:, k]

            y_in = y_prev.detach() if self.detach_y_prev else y_prev

            # Teacher forcing: every tf_every-th step, overwrite observed channels with truth at k-1.
            tf_fires = (k > 0 and k % tf_every == 0)
            if teacher_forcing and tf_fires and y_seq is not None:
                if use_partial:
                    y_in = y_prev.clone()
                    idx = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].to(dtype=y_in.dtype).detach()
                else:
                    y_in = y_seq[:, k - 1, :].to(dtype=y_prev.dtype).detach()

            # u feature already preselected/expanded above.
            u_gru_k_feat = u_gru_k

            # y feature: select GRU y columns, then sqrt_clamp1 (clamp to 1 before sqrt).
            y_in_feat = torch.index_select(y_in, dim=1, index=self.gru_y_idx)
            if y_transform == "sqrt_clamp1":
                y_in_feat = y_in_feat.clamp_min(1.0).sqrt()
            elif y_transform == "sqrt":
                y_in_feat = y_in_feat.clamp_min(1e-6).sqrt()
            elif y_transform == "log1p":
                y_in_feat = torch.log1p(y_in_feat.clamp_min(0.0))

            feat_parts = [u_gru_k_feat, y_in_feat]
            if self.encoder_use_time:
                tau_k = torch.full(
                    (u_gru_k_feat.shape[0], 1),
                    float(k) / float(max(K - 1, 1)),
                    device=u_gru_k_feat.device, dtype=u_gru_k_feat.dtype,
                )
                feat_parts.append(tau_k)
            if self.encoder_use_log_dt:
                # dt-awareness for the variable (OLD 60s vs NEW ~600s) time grid.
                log_dt_k = torch.log(dt_k.clamp_min(1e-6)).to(dtype=u_gru_k_feat.dtype).unsqueeze(-1)
                feat_parts.append(log_dt_k)
            feat = torch.cat(feat_parts, dim=-1)

            x = self.lift(feat).unsqueeze(1)
            z, h = self.gru(x, h)
            raw = self.head(z.squeeze(1))

            # Squash logits into theta bounds (log_gamma, tau = 2.3).
            if self.theta_bounded:
                if self.theta_head_transform == "gamma":
                    theta_k = gamma(raw, self.theta_lo_vec, self.theta_hi_vec)
                else:
                    theta_k = log_gamma(raw, self.theta_lo_vec, self.theta_hi_vec, tau=self.theta_head_tau)
            else:
                theta_k = F.softplus(raw)

            # Bolus jump, then RK4 integrate the mechanism over dt_k.
            y = y_prev + (u_k @ self.u_to_y_jump)
            y = self._rk4_substeps(y, dt_k, theta_k)

            y_out[:, k, :] = y
            th_out[:, k, :] = theta_k
            y_prev = y

        return y_out, th_out


# --- 3. Data loading + split (variable-length; padded steps masked everywhere) ---
def build_u_to_y_jump(control_indices, obs_indices, device=None, dtype=torch.float32):
    """Map a bolus on control column j onto the observed/mechanistic state.
    Identical to jumps.make_u_to_y_jump (P == P_obs case for M5)."""
    c = torch.as_tensor(control_indices, device=device)
    o = torch.as_tensor(obs_indices, device=device)
    U, P = int(c.shape[0]), int(o.shape[0])
    obs_pos = {int(o[p].item()): p for p in range(P)}
    J = torch.zeros((U, P), dtype=dtype, device=c.device)
    for j in range(U):
        p = obs_pos.get(int(c[j].item()), None)
        if p is not None:
            J[j, p] = 1.0
    return J


def _endpoint_values(y_seq, lengths, target_idx):
    """Per-sample final value for each target species: (N, T)."""
    N = y_seq.shape[0]
    end_t = np.clip(lengths.astype(np.int64) - 1, 0, y_seq.shape[1] - 1)
    out = np.empty((N, len(target_idx)), dtype=np.float64)
    for j, t_idx in enumerate(target_idx):
        out[:, j] = y_seq[np.arange(N), end_t, t_idx]
    return out


def _quantile_bin_1d(values, n_bins):
    if n_bins <= 1:
        return np.zeros(values.shape[0], dtype=np.int64)
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(values, q))
    if edges.size <= 2:
        return np.zeros(values.shape[0], dtype=np.int64)
    return np.digitize(values, edges[1:-1], right=True).astype(np.int64)


def _allocate_counts(stratum_sizes, total):
    """Allocate `total` across strata proportionally (largest remainder)."""
    out = np.zeros_like(stratum_sizes, dtype=np.int64)
    if total <= 0 or stratum_sizes.sum() <= 0:
        return out
    raw = total * (stratum_sizes / max(1, int(stratum_sizes.sum())))
    base = np.minimum(np.floor(raw).astype(np.int64), stratum_sizes)
    out[:] = base
    need = int(total - out.sum())
    if need <= 0:
        return out
    frac = raw - np.floor(raw)
    for i in np.argsort(-frac):
        if need <= 0:
            break
        if out[i] < stratum_sizes[i]:
            out[i] += 1
            need -= 1
    if need > 0:
        for i in np.argsort(-stratum_sizes):
            if need <= 0:
                break
            take = min(need, int(stratum_sizes[i] - out[i]))
            if take > 0:
                out[i] += take
                need -= take
    return out


def _stratified_yield_split(pool_idx, y_seq, lengths, n_val, n_test,
                            stratify_bins, stratify_targets, rng):
    """Yield-stratified train/val/test split on a subset of indices.
    Copied from train.py:_stratified_yield_split."""
    if n_test + n_val >= len(pool_idx):
        raise ValueError(f"val={n_val}+test={n_test} >= pool size {len(pool_idx)}")
    sub_y = y_seq[pool_idx]
    sub_L = lengths[pool_idx]
    vals = _endpoint_values(sub_y, sub_L, stratify_targets)
    std = vals.std(axis=0)
    keep = std > 1e-12
    if not np.any(keep):
        shuffled = pool_idx.copy(); rng.shuffle(shuffled)
        return shuffled[n_test + n_val:], shuffled[n_test:n_test + n_val], shuffled[:n_test]
    vals = vals[:, keep]
    ranks = np.empty_like(vals, dtype=np.float64)
    for j in range(vals.shape[1]):
        order = np.argsort(vals[:, j], kind="mergesort")
        inv = np.empty_like(order)
        inv[order] = np.arange(vals.shape[0])
        ranks[:, j] = inv / max(1, vals.shape[0] - 1)
    score = ranks.mean(axis=1)
    labels = _quantile_bin_1d(score, int(stratify_bins)).astype(np.int64)
    strata = []
    for u in np.unique(labels):
        idx_u = pool_idx[labels == u]
        rng.shuffle(idx_u)
        strata.append(idx_u)
    sizes = np.array([len(s) for s in strata], dtype=np.int64)
    take_test = _allocate_counts(sizes, int(n_test))
    test_parts, rem_parts = [], []
    for s, k in zip(strata, take_test):
        test_parts.append(s[:k]); rem_parts.append(s[k:])
    rem_sizes = np.array([len(r) for r in rem_parts], dtype=np.int64)
    take_val = _allocate_counts(rem_sizes, int(n_val))
    val_parts, train_parts = [], []
    for r, k in zip(rem_parts, take_val):
        val_parts.append(r[:k]); train_parts.append(r[k:])
    test_idx = np.concatenate(test_parts) if test_parts else np.empty(0, dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.int64)
    train_idx = np.concatenate(train_parts) if train_parts else np.empty(0, dtype=np.int64)
    return train_idx, val_idx, test_idx


def make_split(cfg: Config, y_seq, lengths, source_idx):
    """Source-aware, yield-stratified train/val/test split.
    Reproduces train.py's stratify_by_source path: OLD and NEW pools are split
    independently via the yield-bin logic, with counts allocated proportional
    to each source's pool size, then concatenated."""
    rng = np.random.default_rng(int(cfg.split_seed))
    targets = [int(t) for t in cfg.stratify_targets if 0 <= int(t) < y_seq.shape[-1]]
    sources_present = sorted(int(s) for s in np.unique(source_idx) if int(s) >= 0)
    assert len(sources_present) >= 2, "stratify_by_source needs >=2 sources"

    pools = {s: np.where(source_idx == s)[0] for s in sources_present}
    sizes_arr = np.array([len(pools[s]) for s in sources_present], dtype=np.int64)
    take_test_per_src = _allocate_counts(sizes_arr, int(cfg.test_n))
    take_val_per_src = _allocate_counts(sizes_arr, int(cfg.val_n))
    tr_parts, va_parts, te_parts = [], [], []
    src_name = {0: "old", 1: "new", 2: "synth"}
    for src, n_te, n_va in zip(sources_present, take_test_per_src, take_val_per_src):
        pool = pools[src]
        tr_s, va_s, te_s = _stratified_yield_split(
            pool, y_seq, lengths, int(n_va), int(n_te), cfg.stratify_bins, targets, rng)
        tr_parts.append(tr_s); va_parts.append(va_s); te_parts.append(te_s)
        print(f"  stratify_by_source[{src_name.get(src, src)}]: pool={len(pool)} "
              f"-> train={len(tr_s)} val={len(va_s)} test={len(te_s)}")
    train_idx = np.concatenate(tr_parts)
    val_idx = np.concatenate(va_parts)
    test_idx = np.concatenate(te_parts)
    rng.shuffle(test_idx); rng.shuffle(val_idx); rng.shuffle(train_idx)
    return train_idx, val_idx, test_idx


class M5Dataset:
    """Loads the npz and exposes per-sample, length-trimmed tensors.
    Mirrors train.py:ODEDataset for this dataset (variable-length, real-only)."""

    def __init__(self, npz_path):
        d = np.load(str(npz_path), allow_pickle=True)
        self.y0 = d["y0"].astype(np.float32)                 # (N, 7)
        self.u_seq = d["u_seq"].astype(np.float32)           # (N, K, 13)
        self.y_seq = d["y_seq"].astype(np.float32)           # (N, K, 7)
        self.dt_per_sample = d["dt_per_sample"].astype(np.float32)  # (N, K)
        self.lengths = d["lengths"].astype(np.int64)         # (N,)
        self.control_indices = d["control_indices"].astype(np.int64)
        self.obs_indices = d["obs_indices"].astype(np.int64)
        # source_label -> int: old=0, new=1, synth=2.
        _src_to_int = {"old": 0, "new": 1, "synth": 2}
        self.source_idx = np.array(
            [_src_to_int.get(str(s), -1) for s in d["source_label"]], dtype=np.int64)
        self.N = self.y0.shape[0]
        self.U = self.u_seq.shape[-1]
        self.P = self.y0.shape[-1]

    def __len__(self):
        return self.N

    def get(self, i):
        """Return (y0, u, y, dt, length) trimmed to the sample's valid length."""
        L = int(self.lengths[i])
        return (
            torch.from_numpy(self.y0[i]),
            torch.from_numpy(self.u_seq[i, :L]),
            torch.from_numpy(self.y_seq[i, :L]),
            torch.from_numpy(np.ascontiguousarray(self.dt_per_sample[i, :L])),
            L,
        )


def collate(ds: M5Dataset, indices):
    """Pad a batch of samples to its own max length; return a lengths tensor.
    Mirrors train.py:collate_varlen (pad_sequence with 0)."""
    y0_l, u_l, y_l, dt_l, src_l = [], [], [], [], []
    for i in indices:
        y0, u, y, dt, L = ds.get(i)
        y0_l.append(y0); u_l.append(u); y_l.append(y); dt_l.append(dt)
        src_l.append(int(ds.source_idx[i]))
    lengths = torch.tensor([u.shape[0] for u in u_l], dtype=torch.long)
    y0 = torch.stack(y0_l)
    u = torch.nn.utils.rnn.pad_sequence(u_l, batch_first=True)
    y = torch.nn.utils.rnn.pad_sequence(y_l, batch_first=True)
    dt = torch.nn.utils.rnn.pad_sequence(dt_l, batch_first=True)
    src = torch.tensor(src_l, dtype=torch.long)
    return y0, u, y, dt, lengths, src


# --- 4. Loss + helpers ---
def build_loss_mask(lengths, K, device):
    """(B, K) bool mask: True for valid (non-padded) timesteps."""
    return torch.arange(K, device=device).unsqueeze(0) < lengths.unsqueeze(1)


def apply_channel_min_gate(y0, y_seq, cols, lengths):
    """Per-sample, per-channel min-subtraction ("gate the observation").
    Lifts each (sample, channel) minimum to ~0 so baseline/failure traces sit
    near zero. Padded steps are masked out of the min. Copied from train.py."""
    B, K, P = y_seq.shape
    ar = torch.arange(K, device=y_seq.device).unsqueeze(0)
    valid = ar < lengths.to(y_seq.device).unsqueeze(1)
    mask = valid.unsqueeze(-1)
    masked_y = torch.where(mask, y_seq, torch.full_like(y_seq, float("inf")))
    idx = torch.as_tensor(cols, device=y_seq.device, dtype=torch.long)
    ch_min = masked_y.index_select(dim=2, index=idx).amin(dim=1, keepdim=True)
    y0_out = y0.clone()
    y_seq_out = y_seq.clone()
    y0_out[:, idx] = y0[:, idx] - ch_min[:, 0, :]
    y_seq_out[:, :, idx] = y_seq[:, :, idx] - ch_min
    return y0_out, y_seq_out


def loss_fn(pred, y_seq, lengths, clamp_min, sample_weights):
    """Masked MSE in log1p space; clamp_min=1.0 floors values so sub-unit mismatches don't count."""
    pred = torch.log1p(pred.clamp_min(clamp_min))
    y_seq = torch.log1p(y_seq.clamp_min(clamp_min))
    se = (pred - y_seq).pow(2)  # (B, K, P)

    mask = build_loss_mask(lengths, se.shape[1], se.device)
    w = mask.unsqueeze(-1).to(se.dtype).expand_as(se)
    if sample_weights is not None:
        w = w.contiguous().clone() * sample_weights.view(-1, 1, 1).to(se.dtype)
    return (se * w).sum() / w.sum().clamp_min(1.0)


# --- 5. R^2 eval: protein at last valid step, mRNA peak; raw units, across test samples ---
def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


@torch.no_grad()
def evaluate_r2(model, ds: M5Dataset, indices, cfg: Config, device, obs_idx):
    """Open-loop rollout over `indices`; return (protein_R2, mRNA_R2).
    m_idx/p_idx are the obs_idx convention [mRNA, protein] = [3, 5]."""
    model.eval()
    m_idx, p_idx = int(cfg.obs_idx[0]), int(cfg.obs_idx[1])
    true_final, pred_final, true_max, pred_max = [], [], [], []
    for i in indices:
        y0, u, y, dt, L = ds.get(i)
        y0_b = y0.unsqueeze(0).to(device)
        u_b = u.unsqueeze(0).to(device)
        y_b = y.unsqueeze(0).to(device)
        dt_b = dt.unsqueeze(0).to(device)
        lengths_b = torch.tensor([L], device=device, dtype=torch.long)
        # Match training: subtract per-sample channel min before the forward pass.
        if cfg.subtract_channel_min:
            y0_b, y_b = apply_channel_min_gate(y0_b, y_b, cfg.subtract_channel_min_cols, lengths_b)
        # Open-loop (teacher_forcing=False).
        pred, _ = model(y0_b, u_b, dt_b, obs_idx, y_seq=None,
                        teacher_forcing=False, y_transform=cfg.y_transform)
        y_np = y_b[0].cpu().numpy()
        p_np = pred[0].cpu().numpy()
        Li = max(1, min(L, y_np.shape[0], p_np.shape[0]))
        true_final.append(y_np[Li - 1, p_idx])
        pred_final.append(p_np[Li - 1, p_idx])
        true_max.append(y_np[:Li, m_idx].max())
        pred_max.append(p_np[:Li, m_idx].max())
    return (r2(true_final, pred_final), r2(true_max, pred_max))


# --- 6. Training loop ---
def device_auto():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(cfg: Config, data_path: Path):
    # --- determinism ---
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = device_auto()
    print(f"Using device: {device}")

    ds = M5Dataset(data_path)
    print(f"Data: N={ds.N}  U={ds.U}  P={ds.P}  "
          f"(old={int((ds.source_idx == 0).sum())}, new={int((ds.source_idx == 1).sum())})")

    # --- split ---
    train_idx, val_idx, test_idx = make_split(cfg, ds.y_seq, ds.lengths, ds.source_idx)
    print(f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # --- scaffold + jump matrix + model ---
    scaffold = M5Scaffold()
    u_to_y_jump = build_u_to_y_jump(ds.control_indices, ds.obs_indices, device=device)
    model = OdeRNN(U=ds.U, rhs=scaffold, u_to_y_jump=u_to_y_jump, cfg=cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: OdeRNN(M5)  hidden={cfg.hidden} layers={cfg.num_layers}  params={n_params:,}")

    obs_idx = torch.tensor(cfg.obs_idx, device=device, dtype=torch.long)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    # LR schedule: linear warmup over warmup_epochs (no cosine decay for M5).
    scheduler = None
    if cfg.warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-6, end_factor=1.0, total_iters=int(cfg.warmup_epochs))
        print(f"LR warmup: {cfg.warmup_epochs} epochs ({cfg.lr:.2e} target)")

    # source_loss_weights -> per-sample multiplier (here all 1.0, kept faithful).
    idx_to_name = {0: "old", 1: "new", 2: "synth"}
    name_to_w = {k: float(v) for k, v in cfg.source_loss_weights.items()}

    # A torch Generator drives epoch-level train shuffling deterministically.
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)

    best_val = float("inf")
    best_state = None

    for ep in range(1, cfg.epochs + 1):
        teacher_forcing = bool(cfg.teacher_forcing) and (ep < int(cfg.tf_drop_epoch))

        # ---- train ----
        model.train()
        perm = torch.randperm(len(train_idx), generator=gen).numpy()
        tr_order = train_idx[perm]
        tr_total, tr_batches = 0.0, 0
        for b0 in range(0, len(tr_order), cfg.batch_size):
            batch_ids = tr_order[b0:b0 + cfg.batch_size]
            y0, u_seq, y_seq, dt_seq, batch_lengths, src_batch = collate(ds, batch_ids)
            y0 = y0.to(device); u_seq = u_seq.to(device)
            y_seq = y_seq.to(device); dt_seq = dt_seq.to(device)
            batch_lengths = batch_lengths.to(device)

            if cfg.subtract_channel_min:
                y0, y_seq = apply_channel_min_gate(
                    y0, y_seq, cfg.subtract_channel_min_cols, batch_lengths)

            opt.zero_grad(set_to_none=True)

            sample_w = torch.tensor(
                [name_to_w.get(idx_to_name.get(int(s), ""), 1.0) for s in src_batch.tolist()],
                device=device, dtype=torch.float32)

            pred, theta = model(
                y0, u_seq, dt_seq, obs_idx, y_seq,
                teacher_forcing=teacher_forcing, tf_every=int(cfg.tf_every),
                y_transform=cfg.y_transform)
            pred = pred[:, :, obs_idx]
            y_sup = y_seq[:, :, obs_idx]

            loss = loss_fn(pred, y_sup, batch_lengths,
                           clamp_min=float(cfg.loss_clamp_min), sample_weights=sample_w)
            loss = loss / float(cfg.loss_normalizer_channels)

            loss.backward()
            # Global grad-norm clip; skip the step if grads are non-finite.
            max_norm = float(cfg.grad_clip) if cfg.grad_clip > 0 else float("inf")
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if not torch.isfinite(total_norm):
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            tr_total += float(loss.item()); tr_batches += 1

        tr_loss = tr_total / max(1, tr_batches)
        if scheduler is not None:
            scheduler.step()

        # ---- val (open-loop, no teacher forcing) ----
        model.eval()
        va_total, va_batches = 0.0, 0
        with torch.no_grad():
            for b0 in range(0, len(val_idx), cfg.batch_size):
                batch_ids = val_idx[b0:b0 + cfg.batch_size]
                y0, u_seq, y_seq, dt_seq, batch_lengths, _ = collate(ds, batch_ids)
                y0 = y0.to(device); u_seq = u_seq.to(device)
                y_seq = y_seq.to(device); dt_seq = dt_seq.to(device)
                batch_lengths = batch_lengths.to(device)
                if cfg.subtract_channel_min:
                    y0, y_seq = apply_channel_min_gate(
                        y0, y_seq, cfg.subtract_channel_min_cols, batch_lengths)
                pred, _ = model(y0, u_seq, dt_seq, obs_idx, y_seq=None,
                                teacher_forcing=False, y_transform=cfg.y_transform)
                pred = pred[:, :, obs_idx]
                y_sup = y_seq[:, :, obs_idx]
                loss = loss_fn(pred, y_sup, batch_lengths,
                               clamp_min=float(cfg.loss_clamp_min), sample_weights=None)
                loss = loss / float(cfg.loss_normalizer_channels)
                va_total += float(loss.item()); va_batches += 1
        va_loss = va_total / max(1, va_batches)

        # Keep the best-by-val weights (the reported model is the val-best one).
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 10 == 0 or ep == cfg.epochs:
            lr_now = opt.param_groups[0]["lr"]
            print(f"epoch {ep:3d}/{cfg.epochs}  train={tr_loss:.5f}  val={va_loss:.5f}  "
                  f"lr={lr_now:.2e}  tf={int(teacher_forcing)}")

    # Restore best-by-val weights for the final test evaluation.
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- final test R^2 ----
    r2_protein, r2_mrna = evaluate_r2(model, ds, test_idx.tolist(), cfg, device, obs_idx)
    print()
    print("=" * 56)
    print(f"TEST endpoint protein R^2 : {r2_protein:.4f}   (target ~0.66)")
    print(f"TEST mRNA-max     R^2     : {r2_mrna:.4f}   (target ~0.24)")
    print("=" * 56)
    return r2_protein, r2_mrna


# --- 7b. Eval the published checkpoint (no training) ---
# Reproduces the paper numbers from the bundled best-val model.pt + saved split.npz.
@torch.no_grad()
def eval_checkpoint(cfg: Config, data_path: Path, ckpt_path: Path, split_path: Path, device):
    ds = M5Dataset(data_path)
    scaffold = M5Scaffold()
    u_to_y_jump = build_u_to_y_jump(ds.control_indices, ds.obs_indices, device=device)
    model = OdeRNN(U=ds.U, rhs=scaffold, u_to_y_jump=u_to_y_jump, cfg=cfg).to(device)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    msd = model.state_dict()
    load = {k: v for k, v in state.items() if k in msd and msd[k].shape == v.shape}
    model.load_state_dict(load, strict=False)
    print(f"Loaded {len(load)}/{len(msd)} tensors from {ckpt_path.name}")

    sp = np.load(str(split_path), allow_pickle=True)
    test_idx = np.asarray(sp[[k for k in sp.files if "test" in k][0]]).tolist()
    obs_idx = torch.tensor(cfg.obs_idx, device=device, dtype=torch.long)
    m_idx, p_idx = int(cfg.obs_idx[0]), int(cfg.obs_idx[1])

    model.eval()
    tf, pf, tm, pm, src = [], [], [], [], []
    for i in test_idx:
        y0, u, y, dt, L = ds.get(i)
        y0b, ub, yb, dtb = (t.unsqueeze(0).to(device) for t in (y0, u, y, dt))
        Lb = torch.tensor([L], device=device, dtype=torch.long)
        if cfg.subtract_channel_min:
            y0b, yb = apply_channel_min_gate(y0b, yb, cfg.subtract_channel_min_cols, Lb)
        pred, _ = model(y0b, ub, dtb, obs_idx, y_seq=None, teacher_forcing=False, y_transform=cfg.y_transform)
        yn, pn = yb[0].cpu().numpy(), pred[0].cpu().numpy()
        Li = max(1, min(L, yn.shape[0], pn.shape[0]))
        tf.append(yn[Li - 1, p_idx]); pf.append(pn[Li - 1, p_idx])
        tm.append(yn[:Li, m_idx].max()); pm.append(pn[:Li, m_idx].max())
        src.append(int(ds.source_idx[i]))
    tf, pf, tm, pm, src = (np.array(a) for a in (tf, pf, tm, pm, src))
    old, new = src == 0, src == 1
    print("=" * 56)
    print(f"TEST protein R^2  : {r2(tf, pf):.4f}   (published 0.6649)")
    print(f"TEST mRNA-max R^2 : {r2(tm, pm):.4f}   (published 0.2379)")
    print(f"  old (deoxygenated) protein R^2 : {r2(tf[old], pf[old]):.4f}  (n={int(old.sum())})")
    print(f"  new (oxygenated)   protein R^2 : {r2(tf[new], pf[new]):.4f}  (n={int(new.sum())})")
    print("=" * 56)


# --- 7. Entry point ---
def main():
    repo_root = Path(__file__).resolve().parents[2]   # .../theta-lab
    default_data = repo_root / CONFIG.dataset_path
    bundled = Path(__file__).resolve().parent / "checkpoints" / "gru_M5"

    ap = argparse.ArgumentParser(description="Standalone CMVF GRU M5 (run HSa_M5_h400_s2).")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override the number of training epochs (default: 200).")
    ap.add_argument("--data", type=str, default=str(default_data),
                    help="Path to the dataset npz (default: resolved from repo root).")
    ap.add_argument("--eval-checkpoint", nargs="?", const=str(bundled / "model.pt"), default=None,
                    help="Skip training: load this checkpoint (default: bundled published model.pt) "
                         "and print the exact paper R^2. Uses --split for the test set.")
    ap.add_argument("--split", type=str, default=str(bundled / "split.npz"),
                    help="Saved split.npz with the held-out test indices (default: bundled).")
    args = ap.parse_args()

    cfg = CONFIG
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    print(f"Dataset: {data_path}")

    if args.eval_checkpoint is not None:
        print(f"Eval-only: checkpoint={args.eval_checkpoint}")
        eval_checkpoint(cfg, data_path, Path(args.eval_checkpoint), Path(args.split), device_auto())
        return

    print(f"Config: HSa_M5_h400_s2  (epochs={cfg.epochs}, seed={cfg.seed})")
    train(cfg, data_path)


if __name__ == "__main__":
    main()
