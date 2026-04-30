# train.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional
import copy
import math
from pathlib import Path
import re
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

from scaffolds import SCAFFOLDS
from models import MODELS
from jumps import make_u_to_y_jump


class ODEDataset(Dataset):
    """
    Required npz fields:
      y0    : (N,P_obs)
      u_seq : (N,K,U)
      y_seq : (N,K,P_obs)
      t_obs : (K+1,)
      control_indices : (U,)
      obs_indices     : (P_obs,)
    Optional:
      names_full, control_names, obs_names
      lengths : (N,) — per-sample valid sequence length.
                When present, __getitem__ returns trimmed tensors
                and collate_varlen pads each batch to its own max.
    """

    def __init__(self, npz_path: str | Path):
        d = np.load(str(npz_path), allow_pickle=True)

        self.y0 = d["y0"].astype(np.float32)  # (N,P_obs)
        self.u_seq = d["u_seq"].astype(np.float32)  # (N,K,U)
        self.y_seq = d["y_seq"].astype(np.float32)  # (N,K,P_obs)
        t_obs = d["t_obs"].astype(np.float32)  # (K+1,)
        self.dt = np.diff(t_obs).astype(np.float32)  # (K,)

        if "control_indices" not in d or "obs_indices" not in d:
            raise ValueError(
                f"Dataset {npz_path} missing control_indices/obs_indices. Regenerate dataset with metadata."
            )
        self.control_indices = d["control_indices"].astype(np.int64)
        self.obs_indices = d["obs_indices"].astype(np.int64)

        self.names_full = d["names_full"].astype(str) if "names_full" in d else None
        self.control_names = d["control_names"].astype(str) if "control_names" in d else None
        self.obs_names = d["obs_names"].astype(str) if "obs_names" in d else None

        # MinMax stats for u (per channel). u_scaled_cols may be a subset of
        # control columns (e.g. excludes DNA c). Map names back to indices in
        # u_seq's last dim so the model can apply per-channel minmax to the
        # right columns and leave others (DNA c) untouched.
        self.u_scale_max = d["u_scale_max"].astype(np.float32) if "u_scale_max" in d else None
        if "u_scaled_cols" in d and self.control_names is not None:
            scaled = list(d["u_scaled_cols"].astype(str))
            ctrl_list = list(self.control_names)
            try:
                self.u_scaled_cols_idx = np.array([ctrl_list.index(c) for c in scaled], dtype=np.int64)
            except ValueError:
                self.u_scaled_cols_idx = None
        else:
            self.u_scaled_cols_idx = None

        # Variable-length support
        if "lengths" in d:
            self.lengths = d["lengths"].astype(np.int64)  # (N,)
            self.variable_length = True
        else:
            self.lengths = None
            self.variable_length = False

    def __len__(self) -> int:
        return self.y0.shape[0]

    def __getitem__(self, i: int):
        if self.variable_length:
            L = int(self.lengths[i])
            return (
                torch.from_numpy(self.y0[i]),          # (P_obs,)
                torch.from_numpy(self.u_seq[i, :L]),   # (L,U)
                torch.from_numpy(self.y_seq[i, :L]),   # (L,P_obs)
            )
        return (
            torch.from_numpy(self.y0[i]),  # (P_obs,)
            torch.from_numpy(self.u_seq[i]),  # (K,U)
            torch.from_numpy(self.y_seq[i]),  # (K,P_obs)
        )


def collate(batch):
    y0, u, y = zip(*batch)
    return torch.stack(y0), torch.stack(u), torch.stack(y), None


def collate_varlen(batch):
    """Pad each batch to its own max length; return lengths tensor."""
    y0_list, u_list, y_list = zip(*batch)
    lengths = torch.tensor([u.shape[0] for u in u_list], dtype=torch.long)
    y0 = torch.stack(y0_list)
    u_padded = torch.nn.utils.rnn.pad_sequence(u_list, batch_first=True)   # (B, K_batch, U)
    y_padded = torch.nn.utils.rnn.pad_sequence(y_list, batch_first=True)   # (B, K_batch, P)
    return y0, u_padded, y_padded, lengths


def _build_loss_mask(lengths: torch.Tensor, K: int, device: torch.device) -> torch.Tensor:
    """Build (B, K) boolean mask: True for valid timesteps."""
    return torch.arange(K, device=device).unsqueeze(0) < lengths.unsqueeze(1)


def _apply_obs_norm(
    y: torch.Tensor,
    method: str,
    stats: Optional[dict] = None,
    is_target: bool = False,
) -> torch.Tensor:
    """Transform y by `method` so loss is computed on the normalized scale.

    Stats dict (when needed) holds float32 (P_obs,) tensors:
      mean, std, min, max — fit on the train split only.

    is_target: True for the ground-truth tensor, False for predictions. Only
    matters for log1p_clip1, which clamps targets at 1 (so failure cases
    y≈0 land at log1p(1)=ln(2), not 0) but leaves predictions clamped at 0
    so gradients still flow when pred<1.
    """
    if method == "none":
        return y
    if method == "sqrt":
        return torch.sqrt(torch.clamp(y, min=0.0))
    if method == "log1p":
        return torch.log1p(torch.clamp(y, min=0.0))
    if method == "log1p_clip1":
        floor = 1.0 if is_target else 0.0
        return torch.log1p(torch.clamp(y, min=floor))
    if method == "zscore":
        m = stats["mean"].to(device=y.device, dtype=y.dtype)
        s = stats["std"].to(device=y.device, dtype=y.dtype)
        return (y - m) / s
    if method == "minmax":
        lo = stats["min"].to(device=y.device, dtype=y.dtype)
        hi = stats["max"].to(device=y.device, dtype=y.dtype)
        return (y - lo) / torch.clamp(hi - lo, min=1e-8)
    raise ValueError(f"Unknown obs_normalization: {method!r}")


def loss_fn(
    pred: torch.Tensor,
    y_seq: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    loss_type: str = "log_mse",
    species_weights: Optional[torch.Tensor] = None,
    obs_normalization: str = "none",
    obs_norm_stats: Optional[dict] = None,
) -> torch.Tensor:
    """Compute per-species loss, optionally masking padded timesteps.

    loss_type:
      - 'log_mse': MSE in log1p space (default, legacy behaviour).
      - 'mse'    : MSE in raw space (or in obs_normalization-transformed space).
      - 'rmse'   : sqrt(MSE) in raw / transformed space.

    obs_normalization (independent of loss_type, applied first):
      - 'none', 'sqrt', 'log1p', 'log1p_clip1', 'zscore', 'minmax'
      - For zscore/minmax, obs_norm_stats must contain (P_obs,) tensors
        mean/std and min/max fit on the train split.
      - Combining with log_mse is unusual (zscore-y can be negative); pair
        non-trivial obs_normalization with loss_type='mse' or 'rmse'.

    species_weights: (P,) tensor applied multiplicatively on the per-species
    squared error before reduction. Typically set to 1 / mean_per_species so
    species with different dynamic ranges contribute comparably.
    """
    if obs_normalization != "none":
        pred = _apply_obs_norm(pred, obs_normalization, obs_norm_stats, is_target=False)
        y_seq = _apply_obs_norm(y_seq, obs_normalization, obs_norm_stats, is_target=True)

    if loss_type == "log_mse":
        se = (torch.log1p(pred) - torch.log1p(y_seq)).pow(2)
    elif loss_type in ("mse", "rmse"):
        se = (pred - y_seq).pow(2)
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}; expected 'log_mse', 'mse', or 'rmse'.")

    if species_weights is not None:
        se = se * species_weights.view(1, 1, -1)

    if lengths is not None:
        mask = _build_loss_mask(lengths, se.shape[1], se.device)  # (B,K)
        se = se * mask.unsqueeze(-1)
        denom = mask.sum() * se.shape[-1]
        out = se.sum() / denom
    else:
        out = se.mean()

    if loss_type == "rmse":
        out = torch.sqrt(out + 1e-12)
    return out


def _build_tail_mask(
    lengths: Optional[torch.Tensor],
    K: int,
    device: torch.device,
    frac: float,
    batch_size: int,
) -> torch.Tensor:
    """Boolean (B, K) mask selecting the final `frac` of each sample's valid length."""
    if lengths is None:
        lens = torch.full((batch_size,), K, device=device, dtype=torch.long)
    else:
        lens = lengths
    tail = torch.clamp((lens.float() * frac).long(), min=1)
    idx = torch.arange(K, device=device).unsqueeze(0)  # (1,K)
    start = (lens - tail).unsqueeze(1)                 # (B,1)
    end = lens.unsqueeze(1)                            # (B,1)
    return (idx >= start) & (idx < end)


def r_terminal_loss(
    pred_full: torch.Tensor,
    r_idx: int,
    lengths: Optional[torch.Tensor],
    frac: float = 0.05,
) -> torch.Tensor:
    """Force R -> 0 over the final `frac` of each trajectory (target column is zeros)."""
    B, K = pred_full.shape[0], pred_full.shape[1]
    r = pred_full[:, :, r_idx]  # (B,K)
    mask = _build_tail_mask(lengths, K, pred_full.device, frac, B).to(r.dtype)
    se = (r ** 2) * mask
    denom = mask.sum().clamp(min=1.0)
    return se.sum() / denom


def pm_tail_loss(
    pred: torch.Tensor,
    y_true: torch.Tensor,
    lengths: Optional[torch.Tensor],
    pm_obs_idx: int,
    frac: float = 0.25,
    obs_normalization: str = "none",
    obs_norm_stats: Optional[dict] = None,
) -> torch.Tensor:
    """MSE on protein over the last `frac` of each trajectory.

    Per supervisor: 'progressively weight the tail of the simulation for protein
    heavier ... focuses on the final yield more'. Forces the model to land its
    final pm prediction precisely, including for failure cases (pm≈0 throughout).
    """
    if obs_normalization != "none":
        pred = _apply_obs_norm(pred, obs_normalization, obs_norm_stats, is_target=False)
        y_true = _apply_obs_norm(y_true, obs_normalization, obs_norm_stats, is_target=True)
    B, K, _ = pred.shape
    mask = _build_tail_mask(lengths, K, pred.device, frac, B).to(pred.dtype)  # (B, K)
    pm_pred = pred[:, :, pm_obs_idx]
    pm_true = y_true[:, :, pm_obs_idx]
    se = (pm_pred - pm_true).pow(2) * mask
    return se.sum() / mask.sum().clamp(min=1.0)


def endpoint_loss(
    pred: torch.Tensor,
    y_true: torch.Tensor,
    lengths: Optional[torch.Tensor],
    mrna_obs_idx: int,
    protein_obs_idx: int,
    species_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """log_mse on per-trajectory endpoints: protein-final and mRNA-peak.

    Forces the model to differentiate experiments by their *outcomes*, which
    per-step MSE alone doesn't punish (the per-step loss is happily minimised
    by predicting the dataset mean trajectory).

    species_weights, if provided, must be the same (P_obs,) vector used by the
    per-step loss (e.g. 1/mean). Endpoint terms are weighted by the matching
    indices so this term lives on the same scale as the per-step loss — λ=1
    then means "endpoint loss has equal weight to the trajectory loss".
    """
    B, K, _ = pred.shape
    if lengths is None:
        end_idx = torch.full((B,), K - 1, device=pred.device, dtype=torch.long)
        valid_mask = torch.ones(B, K, device=pred.device, dtype=pred.dtype)
    else:
        lens = lengths.to(pred.device, dtype=torch.long)
        end_idx = (lens - 1).clamp_min(0)
        idx = torch.arange(K, device=pred.device).unsqueeze(0)
        valid_mask = (idx < lens.unsqueeze(1)).to(pred.dtype)

    b_idx = torch.arange(B, device=pred.device)
    pred_pm = pred[b_idx, end_idx, protein_obs_idx]
    true_pm = y_true[b_idx, end_idx, protein_obs_idx]

    # max(pred[:, :L_i, mm]) per trajectory, masking past lengths.
    NEG_INF = torch.finfo(pred.dtype).min
    pred_mm_seq = pred[:, :, mrna_obs_idx].masked_fill(valid_mask < 0.5, NEG_INF)
    true_mm_seq = y_true[:, :, mrna_obs_idx].masked_fill(valid_mask < 0.5, NEG_INF)
    pred_mm_peak = pred_mm_seq.max(dim=1).values
    true_mm_peak = true_mm_seq.max(dim=1).values

    pm_term = (torch.log1p(pred_pm.clamp_min(0)) - torch.log1p(true_pm.clamp_min(0))).pow(2).mean()
    mm_term = (torch.log1p(pred_mm_peak.clamp_min(0)) - torch.log1p(true_mm_peak.clamp_min(0))).pow(2).mean()

    if species_weights is not None:
        w_mm = species_weights[mrna_obs_idx]
        w_pm = species_weights[protein_obs_idx]
        # Match per-step loss which sums w_i * MSE_i over species, then divides by 2 (P_obs).
        return 0.5 * (w_mm * mm_term + w_pm * pm_term)
    return 0.5 * (pm_term + mm_term)


def loss_fn_per_species(
    pred: torch.Tensor,
    y_seq: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    loss_type: str = "log_mse",
    species_weights: Optional[torch.Tensor] = None,
    obs_normalization: str = "none",
    obs_norm_stats: Optional[dict] = None,
) -> torch.Tensor:
    if obs_normalization != "none":
        pred = _apply_obs_norm(pred, obs_normalization, obs_norm_stats, is_target=False)
        y_seq = _apply_obs_norm(y_seq, obs_normalization, obs_norm_stats, is_target=True)

    if loss_type == "log_mse":
        se = (torch.log1p(pred) - torch.log1p(y_seq)).pow(2)
    elif loss_type in ("mse", "rmse"):
        se = (pred - y_seq).pow(2)
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    if species_weights is not None:
        se = se * species_weights.view(1, 1, -1)

    if lengths is not None:
        mask = _build_loss_mask(lengths, se.shape[1], se.device)
        se = se * mask.unsqueeze(-1)
        out = se.sum(dim=(0, 1)) / mask.sum()
    else:
        out = se.mean(dim=(0, 1))

    if loss_type == "rmse":
        out = torch.sqrt(out + 1e-12)
    return out


def compute_species_mean_weights(
    y_seq: np.ndarray,
    lengths: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Per-species weight = 1 / mean_over_valid_timesteps(y_seq).

    Returns a (P,) float32 vector. Intended for normalizing species that
    differ in dynamic range (e.g. mRNA vs protein) so each contributes
    comparably to the loss.
    """
    y = np.asarray(y_seq, dtype=np.float64)  # (N,K,P)
    N, K, P = y.shape
    if lengths is None:
        means = y.reshape(-1, P).mean(axis=0)
    else:
        lens = np.asarray(lengths, dtype=np.int64)
        mask = (np.arange(K)[None, :] < lens[:, None]).astype(np.float64)  # (N,K)
        num = (y * mask[..., None]).sum(axis=(0, 1))                        # (P,)
        den = mask.sum() + eps
        means = num / den
    return (1.0 / np.maximum(means, eps)).astype(np.float32)


def _endpoint_values(
    y_seq: np.ndarray,
    lengths: Optional[np.ndarray],
    target_idx: list[int],
) -> np.ndarray:
    """Per-sample final values for selected species indices: (N, T)."""
    N = y_seq.shape[0]
    if lengths is None:
        return y_seq[:, -1, target_idx].astype(np.float64)
    end_t = np.clip(lengths.astype(np.int64) - 1, 0, y_seq.shape[1] - 1)
    out = np.empty((N, len(target_idx)), dtype=np.float64)
    for j, t_idx in enumerate(target_idx):
        out[:, j] = y_seq[np.arange(N), end_t, t_idx]
    return out


def _quantile_bin_1d(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile binning robust to ties; returns integer labels."""
    if n_bins <= 1:
        return np.zeros(values.shape[0], dtype=np.int64)
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, q)
    edges = np.unique(edges)
    if edges.size <= 2:
        return np.zeros(values.shape[0], dtype=np.int64)
    return np.digitize(values, edges[1:-1], right=True).astype(np.int64)


def _allocate_counts(stratum_sizes: np.ndarray, total: int) -> np.ndarray:
    """Allocate `total` across strata proportionally (largest remainder)."""
    out = np.zeros_like(stratum_sizes, dtype=np.int64)
    if total <= 0 or stratum_sizes.sum() <= 0:
        return out
    raw = total * (stratum_sizes / max(1, int(stratum_sizes.sum())))
    base = np.floor(raw).astype(np.int64)
    base = np.minimum(base, stratum_sizes)
    out[:] = base

    need = int(total - out.sum())
    if need <= 0:
        return out

    frac = raw - np.floor(raw)
    order = np.argsort(-frac)
    for i in order:
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


def _make_split_indices(
    *,
    N: int,
    y_seq: np.ndarray,
    lengths: Optional[np.ndarray],
    n_val: int,
    n_test: int,
    split_seed: int,
    stratified_split: bool,
    stratify_bins: int,
    stratify_targets: Optional[list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create train/val/test indices; optionally stratified by endpoint bins."""
    if n_test + n_val >= N:
        raise ValueError(f"val_n={n_val} + test_n={n_test} >= N={N}")

    rng = np.random.default_rng(split_seed)
    all_idx = np.arange(N, dtype=np.int64)

    if not stratified_split:
        rng.shuffle(all_idx)
        test_idx = all_idx[:n_test]
        val_idx = all_idx[n_test:n_test + n_val]
        train_idx = all_idx[n_test + n_val:]
        return train_idx, val_idx, test_idx

    P = int(y_seq.shape[-1])
    targets = list(stratify_targets) if stratify_targets is not None else list(range(P))
    targets = [int(t) for t in targets if 0 <= int(t) < P]
    if len(targets) == 0:
        raise ValueError("stratified_split=True but stratify_targets is empty/invalid")

    vals = _endpoint_values(y_seq, lengths, targets)  # (N,T)
    std = vals.std(axis=0)
    keep = std > 1e-12
    if not np.any(keep):
        rng.shuffle(all_idx)
        test_idx = all_idx[:n_test]
        val_idx = all_idx[n_test:n_test + n_val]
        train_idx = all_idx[n_test + n_val:]
        return train_idx, val_idx, test_idx
    vals = vals[:, keep]

    ranks = np.empty_like(vals, dtype=np.float64)
    for j in range(vals.shape[1]):
        order = np.argsort(vals[:, j], kind="mergesort")
        inv = np.empty_like(order)
        inv[order] = np.arange(vals.shape[0])
        denom = max(1, vals.shape[0] - 1)
        ranks[:, j] = inv / denom
    score = ranks.mean(axis=1)
    labels = _quantile_bin_1d(score, int(stratify_bins)).astype(np.int64)

    uniq = np.unique(labels)
    strata = []
    for u in uniq:
        idx_u = np.where(labels == u)[0]
        rng.shuffle(idx_u)
        strata.append(idx_u)

    sizes = np.array([len(s) for s in strata], dtype=np.int64)
    take_test = _allocate_counts(sizes, int(n_test))

    test_parts: list[np.ndarray] = []
    rem_parts: list[np.ndarray] = []
    for s, k in zip(strata, take_test):
        test_parts.append(s[:k])
        rem_parts.append(s[k:])

    rem_sizes = np.array([len(s) for s in rem_parts], dtype=np.int64)
    take_val = _allocate_counts(rem_sizes, int(n_val))

    val_parts: list[np.ndarray] = []
    train_parts: list[np.ndarray] = []
    for s, k in zip(rem_parts, take_val):
        val_parts.append(s[:k])
        train_parts.append(s[k:])

    test_idx = np.concatenate(test_parts) if len(test_parts) else np.empty(0, dtype=np.int64)
    val_idx = np.concatenate(val_parts) if len(val_parts) else np.empty(0, dtype=np.int64)
    train_idx = np.concatenate(train_parts) if len(train_parts) else np.empty(0, dtype=np.int64)

    rng.shuffle(test_idx)
    rng.shuffle(val_idx)
    rng.shuffle(train_idx)
    return train_idx, val_idx, test_idx


@dataclass
class TrainConfig:
    dataset_path: str

    study: str = "adhoc"
    tags: list[str] | None = None
    exp_name: str = "default"
    out_root: str = "experiments"

    save_model_name: str = "model.pt"  # saved in exp_dir/
    save_last_name: str = "model_last.pt"  # saved in exp_dir/
    save_curves_name: str = "loss_curves.npz"  # saved in exp_dir/logs/

    epochs: int = 200
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 0  # linear LR warmup; 0 disables
    cosine_decay: bool = False  # cosine decay from lr to lr*cosine_decay_min after warmup
    val_n: int = 100   # fixed count for validation set
    test_n: int = 100  # fixed count for held-out test set
    # legacy: val_frac still accepted but val_n/test_n take precedence when > 0
    val_frac: float = 0.0
    seed: int = 42        # controls model init + training stochasticity only
    split_seed: int = 42   # controls train/val/test split — keep fixed across seeds

    # Split balancing by endpoint outcomes (final y_seq values).
    # When enabled, bins selected target species by endpoint quantiles and
    # samples train/val/test proportionally from each bin combination.
    stratified_split: bool = False
    stratify_bins: int = 5
    stratify_targets: list[int] | None = None

    num_workers: int = 0
    pin_memory: bool = True

    scaffold: str = "reduced5"
    hidden: int = 128
    lift_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    theta_lo: float = 1e-3
    theta_hi: float = 2.0
    theta_lo_vec: list[float] | None = None
    theta_hi_vec: list[float] | None = None
    n_substeps: int = 1

    ff_mult: int = 2  # feedforward multiplier inside each transformer layer (2=current, 4=standard)

    # Transformer-specific (ignored by non-Transformer models via **kwargs)
    context_len: int = 64  # sliding window size; set >= sequence length for full context
    tf_group_size: int = 32  # grouped-TF chunk size for ode_transformer_grouped
    ar_gap: int = 4  # autoregressive steps inserted between grouped-TF chunks

    # Mamba-specific (ignored by non-Mamba models via **kwargs)
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4

    forget_bias_init: Optional[float] = None  # None = PyTorch default; 1.0 = Gers/Jozefowicz positive shift
    legacy_forget_bias_bug: bool = False      # reproduce pre-fix fill_(0.0) on both bias_ih and bias_hh

    use_basal: bool = False
    beta_regularization: bool = False
    lambda_beta: float = 1.0

    theta_bounded: bool = True   # if False, use softplus (unbounded above) instead of gamma

    grad_clip: float = 1.0
    teacher_forcing: bool = True
    tf_every: int = 50
    tf_drop_epoch: int = 10**9
    u_transform: str = "none"  # GRU input transform: "none" | "cumsum" | "sqrt" | "cumsum_sqrt"
    y_transform: str = "none"  # y_in transform for GRU: "none" | "sqrt" | "log1p". Use sqrt/log1p
                               # to prevent large state values (e.g. protein ~10^3) from dominating
                               # the lift layer over reagent inputs (~1) — root cause of mean-trajectory
                               # collapse on real_ivtt data.
    gru_y_obs_only: bool = False  # if True, feed only obs_idx columns of y to GRU (matches the
                                  # supervisor's reference setup which only feeds mm_prev, pm_prev).
                                  # Latent-state values are model fabrications during open-loop rollout
                                  # and provide no useful signal.
    exclude_ode_cols_from_gru: bool = False  # if True, exclude u cols that route to ODE states (e.g. DNA c)
    head_bias_init: float = 0.0   # init all head biases to this (<0 starts theta near lo; e.g. -5.0)
    head_weight_gain: float = 1.0  # Xavier gain for head weights (>1 amplifies per-experiment variation)
    detach_y_prev: bool = True   # detach y_prev before feeding to GRU (matches supervisor reference).
                                 # If False, gradients flow through y_prev → GRU → theta across timesteps,
                                 # giving a longer credit-assignment path but more memory + instability.

    # Architecture toggles to mirror supervisor reference (datasets/supervisorhint_dataparser.py):
    #   - theta_head_transform: "log_gamma" (default, geometric midpoint at init)
    #                           vs "gamma" (linear-sigmoid, arithmetic midpoint — supervisor)
    #   - head_bottle: insert hidden→120→SiLU→40→SiLU before head (supervisor's `bottle`)
    #   - lift_skip: drop the lift MLP and feed [u_feat, y_feat] straight into GRU
    theta_head_transform: str = "log_gamma"
    head_bottle: bool = False
    lift_skip: bool = False

    # checkpointing cadence (0 disables periodic ckpts)
    ckpt_every: int = 10

    l1_regularization: bool = False   # smoothness: penalizes mean |theta[t] - theta[t-1]|
    l2_regularization: bool = False   # smoothness: penalizes mean (theta[t] - theta[t-1])^2

    lambda_reg: float = 0.001

    # Loss type: 'log_mse' (default, legacy), 'mse', 'rmse'.
    loss_type: str = "log_mse"
    # If True, weight per-species SE by 1/mean(species) computed from the
    # training set (real data scenario). Keeps mRNA/protein comparable.
    normalize_by_species_mean: bool = False

    # Apply a transform to pred AND target before computing the loss.
    # Solves the "log_mse compresses predictions into a narrow scale band"
    # diagnostic finding (Spearman r > Pearson r → ranking right, scale wrong).
    # Stats are fit on the train split only. Saved to norm_stats.npz for
    # inverse transform during plotting / endpoint_r2.
    #   "none"   → no transform (current default)
    #   "sqrt"   → MSE on sqrt(y)            — light scale compression
    #   "log1p"  → MSE on log1p(y)           — equivalent to current log_mse
    #   "zscore" → MSE on (y - mean) / std   — per-species standardised
    #   "minmax" → MSE on (y - min) / (max - min)  — bounded to [0,1]
    obs_normalization: str = "none"

    # Terminal R -> 0 regularization (TXTLResourceandMaturationDNAScaffold).
    # Adds loss pushing the R state to 0 over the final `r_terminal_frac` of
    # each trajectory. Disabled when lambda_r_terminal == 0.
    lambda_r_terminal: float = 0.0
    r_terminal_frac: float = 0.05

    # Auto-rescale every aux loss (r_terminal, o_terminal, endpoint) by
    # main_loss.detach() / aux_loss.detach() before applying its lambda.
    # With this on, lambda has a STABLE cross-config meaning: "aux loss
    # contributes lambda × main_loss to gradient magnitude". Without it,
    # changing obs_normalization or loss_type silently rescales every aux
    # term by orders of magnitude, contaminating the comparison.
    auto_scale_aux_losses: bool = False

    # Terminal O -> 0 regularization (TXTLResourceandMaturationDNAScaffold).
    # Softly encourages oxygen to decay over the final `o_terminal_frac`.
    # Disabled when lambda_o_terminal == 0.
    lambda_o_terminal: float = 0.0
    o_terminal_frac: float = 0.05

    # Endpoint loss: log_mse on per-trajectory protein-final + mRNA-peak.
    # Counters mean-trajectory collapse where per-step MSE alone is happy
    # predicting the dataset mean (real_ivtt diagnose: pred-mm cross-traj std
    # / true std ≈ 0.02). Disabled when lambda_endpoint == 0.
    # Indices are positions in the SUPERVISED obs space (after obs_idx slicing),
    # i.e. for obs_idx=[3,5] -> mrna at 0, protein at 1.
    lambda_endpoint: float = 0.0
    endpoint_mrna_obs_idx: int = 0
    endpoint_protein_obs_idx: int = 1

    # Tail-weighted protein loss (supervisor's "weight the tail of the
    # simulation for protein heavier"). Adds MSE on pm over the last
    # `pm_tail_frac` of each trajectory, with auto-scaling.
    lambda_pm_tail: float = 0.0
    pm_tail_obs_idx: int = 1     # pm position in the supervised obs space
    pm_tail_frac: float = 0.25

    # If set (e.g. [0, 12]), supervise loss/TF only on those species indices.
    # If null/None, supervises all observed species (default behaviour).
    obs_idx: list[int] | None = None

    wandb_enabled: bool = False
    wandb_project: str = "theta-lab"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str = "train"
    wandb_name: str | None = None
    wandb_mode: str | None = None

    jit_scripting: bool = False
    torch_compile: bool = False
    autocast_bf16: bool = False

    endpoint_r2: bool = False  # if True, runs endpoint R² analysis & saves plot in exp_dir

    # Early stopping: stop after `early_stop_patience` epochs without best_val improvement.
    # 0 (default) disables. `early_stop_min_delta` requires at least this much improvement
    # to count as progress (set >0 to ignore tiny noise-level decreases).
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0

    # 'ode_rnn' (default), 'ode_rnn_2020' (latent ODE-RNN style),
    # or 'neural_ode' (pure MLP baseline)
    model_class: str = "ode_rnn"


def load_cfg(path: str | Path) -> TrainConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return TrainConfig(**d)



def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    text = text.strip("-_.")
    return text or "run"


def build_run_dir(cfg: TrainConfig, now: datetime) -> tuple[Path, str, str]:
    study_slug = slugify(cfg.study)
    run_name = slugify(cfg.exp_name)
    run_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{run_name}"
    run_dir = Path(cfg.out_root) / study_slug / run_id
    return run_dir, run_id, study_slug


def init_wandb(cfg: TrainConfig, cfg_dict: dict, *, run_id: str, exp_dir: Path):
    if not cfg.wandb_enabled:
        return None, None

    try:
        import wandb
    except ImportError:
        print("[wandb] wandb is not installed; continuing without W&B logging.")
        return None, None

    raw_tags = cfg.tags or []
    if isinstance(raw_tags, str):
        tags = [raw_tags]
    else:
        tags = [str(tag) for tag in raw_tags]
    if cfg.study not in tags:
        tags.append(str(cfg.study))

    init_kwargs = {
        "project": cfg.wandb_project,
        "entity": cfg.wandb_entity,
        "group": cfg.wandb_group or cfg.study,
        "job_type": cfg.wandb_job_type,
        "name": cfg.wandb_name or run_id,
        "tags": tags,
        "config": cfg_dict,
        "dir": str(exp_dir),
    }
    if cfg.wandb_mode is not None:
        init_kwargs["mode"] = cfg.wandb_mode

    try:
        run = wandb.init(**{k: v for k, v in init_kwargs.items() if v is not None})
    except Exception as exc:
        print(f"[wandb] init failed: {exc}")
        return None, None

    if run is not None:
        run.config.update(
            {
                "run_id": run_id,
                "study": cfg.study,
                "run_dir": str(exp_dir.resolve()),
            },
            allow_val_change=True,
        )
    return wandb, run


def log_wandb_images(wandb, run, plots_dir: Path) -> None:
    if wandb is None or run is None or not plots_dir.exists():
        return

    single_images = [
        ("plots/loss_curves", plots_dir / "loss_curves.png"),
        ("plots/val_species_heatmap", plots_dir / "val_species_heatmap.png"),
        ("plots/val_species_final", plots_dir / "val_species_final.png"),
        ("plots/pred_overlays", plots_dir / "pred_overlays_sample000.png"),
    ]

    payload = {}
    for key, path in single_images:
        if path.exists():
            payload[key] = wandb.Image(str(path))

    pred_paths = sorted(plots_dir.glob("pred_vs_true_*.png"))[:3]
    if pred_paths:
        payload["plots/pred_vs_true_examples"] = [
            wandb.Image(str(path), caption=path.stem) for path in pred_paths
        ]

    theta_paths = sorted(plots_dir.glob("theta_from_pred_*.png"))[:3]
    if theta_paths:
        payload["plots/theta_examples"] = [
            wandb.Image(str(path), caption=path.stem) for path in theta_paths
        ]

    if payload:
        run.log(payload)


def log_wandb_artifact(wandb, run, *, exp_dir: Path, run_id: str) -> None:
    if wandb is None or run is None:
        return

    artifact = wandb.Artifact(run_id, type="experiment")
    for rel_path in [
        Path("config.yaml"),
        Path("model.pt"),
        Path("model_last.pt"),
        Path("logs") / "loss_curves.npz",
    ]:
        path = exp_dir / rel_path
        if path.exists():
            artifact.add_file(str(path), name=str(rel_path))

    try:
        run.log_artifact(artifact)
    except Exception as exc:
        print(f"[wandb] artifact logging failed: {exc}")


def _clean_state_dict(model: nn.Module) -> dict:
    """Strip torch.compile's _orig_mod. prefix so state dicts load cleanly."""
    return {k.removeprefix("_orig_mod."): v.detach().cpu() for k, v in model.state_dict().items()}


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(cfg: TrainConfig, *, no_plot: bool = False, plot_samples: int = 5, plot_sample_idx: int = 0) -> None:
    t0 = time.time()

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = device_auto()
    print(f"Using device: {device}")

    now = datetime.now()
    exp_dir, run_id, study_slug = build_run_dir(cfg, now)
    exp_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = exp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config for later reconstruction
    cfg_dict = asdict(cfg)
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    print(f"Experiment: {exp_dir}")
    print(f"Study: {study_slug} | Run ID: {run_id}")

    wandb, wandb_run = init_wandb(cfg, cfg_dict, run_id=run_id, exp_dir=exp_dir)

    ds = ODEDataset(cfg.dataset_path)
    N = len(ds)

    n_test = int(cfg.test_n) if cfg.test_n > 0 else 0
    n_val  = int(cfg.val_n)  if cfg.val_n  > 0 else max(1, int(N * cfg.val_frac))
    train_idx, val_idx, test_idx = _make_split_indices(
        N=N,
        y_seq=ds.y_seq,
        lengths=ds.lengths if ds.variable_length else None,
        n_val=n_val,
        n_test=n_test,
        split_seed=int(cfg.split_seed),
        stratified_split=bool(cfg.stratified_split),
        stratify_bins=int(cfg.stratify_bins),
        stratify_targets=cfg.stratify_targets,
    )

    if cfg.stratified_split:
        print(
            "Split: stratified"
            f" | bins={int(cfg.stratify_bins)}"
            f" | targets={cfg.stratify_targets if cfg.stratify_targets is not None else 'auto'}"
        )

    # persist split so plotting always uses the correct test indices
    np.savez(exp_dir / "split.npz",
             train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    collate_fn = collate_varlen if ds.variable_length else collate

    train_loader = DataLoader(
        torch.utils.data.Subset(ds, train_idx.tolist()),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=bool(cfg.pin_memory),
    )

    val_loader = None
    if len(val_idx) > 0:
        val_loader = DataLoader(
            torch.utils.data.Subset(ds, val_idx.tolist()),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=bool(cfg.pin_memory),
        )

    # infer dims
    y0_ex, u_ex, _ = ds[0]
    P_obs = int(y0_ex.shape[0])
    U = int(u_ex.shape[-1])

    if cfg.scaffold not in SCAFFOLDS:
        raise ValueError(f"Unknown scaffold '{cfg.scaffold}'. Available: {list(SCAFFOLDS.keys())}")
    scaffold = copy.deepcopy(SCAFFOLDS[cfg.scaffold])

    if cfg.theta_lo_vec is not None or cfg.theta_hi_vec is not None:
        if cfg.theta_lo_vec is None or cfg.theta_hi_vec is None:
            raise ValueError("theta_lo_vec and theta_hi_vec must be set together.")
        if len(cfg.theta_lo_vec) != scaffold.theta_dim or len(cfg.theta_hi_vec) != scaffold.theta_dim:
            raise ValueError(
                f"theta_lo_vec/hi_vec must be length {scaffold.theta_dim} for scaffold {cfg.scaffold}."
            )
        for i, (lo, hi) in enumerate(zip(cfg.theta_lo_vec, cfg.theta_hi_vec)):
            if float(lo) >= float(hi):
                raise ValueError(
                    f"theta bounds invalid at index {i}: lo={lo} must be < hi={hi}."
                )
        scaffold.theta_lo_vec = [float(v) for v in cfg.theta_lo_vec]
        scaffold.theta_hi_vec = [float(v) for v in cfg.theta_hi_vec]
        print(f"Overriding theta bounds from config: lo={scaffold.theta_lo_vec} hi={scaffold.theta_hi_vec}")

    if scaffold.P != P_obs:
        raise ValueError(f"Scaffold {cfg.scaffold} expects P={scaffold.P}, but dataset has P_obs={P_obs}.")

    u_to_y_jump = make_u_to_y_jump(ds.control_indices, ds.obs_indices, device=device)  # (U,P_obs)

    # Columns that route to an ODE state (control_indices[j] < P) are ODE-only;
    # exclude them from GRU input so the GRU only sees reagent boluses.
    gru_u_cols: list[int] | None = None
    if cfg.exclude_ode_cols_from_gru:
        gru_u_cols = [j for j in range(U) if int(ds.control_indices[j]) >= scaffold.P]
        excluded = [j for j in range(U) if int(ds.control_indices[j]) < scaffold.P]
        print(f"GRU u cols: {len(gru_u_cols)}/{U} (excluded ODE-routed cols: {excluded})")

    # When `gru_y_obs_only`, restrict y_in to observed species so the GRU is not
    # confused by latent-state values which are model fabrications during open-loop rollout.
    gru_y_cols: list[int] | None = None
    if bool(cfg.gru_y_obs_only):
        gru_y_cols = list(cfg.obs_idx) if cfg.obs_idx is not None else list(range(int(scaffold.P)))
        print(f"GRU y cols: {gru_y_cols}/{scaffold.P} (obs-only feedback to GRU)")

    if cfg.model_class not in MODELS:
        raise ValueError(f"Unknown model_class '{cfg.model_class}'. Available: {list(MODELS.keys())}")
    model = MODELS[cfg.model_class](
        U=U,
        rhs=scaffold,
        u_to_y_jump=u_to_y_jump,
        hidden=cfg.hidden,
        lift_dim=cfg.lift_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        ff_mult=cfg.ff_mult,
        theta_lo=cfg.theta_lo,
        theta_hi=cfg.theta_hi,
        n_substeps=cfg.n_substeps,
        use_basal=cfg.use_basal,
        context_len=cfg.context_len,
        tf_group_size=cfg.tf_group_size,
        ar_gap=cfg.ar_gap,
        theta_bounded=cfg.theta_bounded,
        d_state=cfg.d_state,
        expand=cfg.expand,
        d_conv=cfg.d_conv,
        forget_bias_init=cfg.forget_bias_init,
        legacy_forget_bias_bug=cfg.legacy_forget_bias_bug,
        gru_u_cols=gru_u_cols,
        gru_y_cols=gru_y_cols,
        head_bias_init=float(cfg.head_bias_init),
        head_weight_gain=float(cfg.head_weight_gain),
        detach_y_prev=bool(cfg.detach_y_prev),
        u_minmax_max=(torch.tensor(ds.u_scale_max, dtype=torch.float32)
                      if str(cfg.u_transform) in ("minmax", "minmax_sqrt") and ds.u_scale_max is not None
                      else None),
        u_minmax_cols=(list(ds.u_scaled_cols_idx)
                       if str(cfg.u_transform) in ("minmax", "minmax_sqrt") and ds.u_scaled_cols_idx is not None
                       else None),
        theta_head_transform=str(cfg.theta_head_transform),
        head_bottle=bool(cfg.head_bottle),
        lift_skip=bool(cfg.lift_skip),
    ).to(device)

    # DIAGNOSTIC ONLY — safe to remove once confirmed Flash Attention works on your GPU.
    if cfg.autocast_bf16 and cfg.model_class in {"ode_transformer", "ode_transformer_grouped"} and device.type == "cuda":
        _B, _W, _H = 2, 8, model.hidden
        _x = torch.randn(_B, _W, _H, device=device, dtype=torch.bfloat16)
        _mask = nn.Transformer.generate_square_subsequent_mask(_W, device=device).to(torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                try:
                    model.transformer(_x, mask=_mask, is_causal=True)
                    print("Flash Attention: OK")
                except Exception as e:
                    print(f"Flash Attention: NOT available ({e})")
        del _x, _mask

    compile_model = cfg.torch_compile
    jit_scripting = cfg.jit_scripting

    if jit_scripting == True:
        try:
            model = torch.jit.script(model)
            print('The model compiled successfully')
        except:
            print('The model did not compile please check')

    elif compile_model == True:
        try:
            # dynamic=True: generates one symbolic kernel per distinct graph instead of
            # recompiling for every new tensor shape. Critical for the OdeTransformer
            # whose attention window W grows 1..context_len during the forward loop —
            # without this, Dynamo recompiles 64 times (~hours on GPFS clusters).
            model = torch.compile(model, dynamic=True)
            print("torch.compile: OK (dynamic=True)")
        except Exception as e:
            print(f'The model did not compile: {e}')


    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    scheduler = None
    if cfg.warmup_epochs > 0 and cfg.cosine_decay:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-6, end_factor=1.0, total_iters=int(cfg.warmup_epochs)
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, cfg.epochs - cfg.warmup_epochs), eta_min=float(cfg.lr) * 0.01
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[int(cfg.warmup_epochs)]
        )
        print(f"LR warmup: {cfg.warmup_epochs} epochs ({cfg.lr:.2e} target) + cosine decay to {float(cfg.lr)*0.01:.2e}")
    elif cfg.warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-6, end_factor=1.0, total_iters=int(cfg.warmup_epochs)
        )
        print(f"LR warmup: {cfg.warmup_epochs} epochs ({cfg.lr:.2e} target)")
    elif cfg.cosine_decay:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs, eta_min=float(cfg.lr) * 0.01
        )
        print(f"LR cosine decay: {cfg.lr:.2e} → {float(cfg.lr)*0.01:.2e} over {cfg.epochs} epochs")

    mech_names = ds.obs_names.tolist() if ds.obs_names is not None else None
    if cfg.obs_idx is not None and mech_names is not None:
        mech_names = [mech_names[i] for i in cfg.obs_idx]

    print(f"Data: N={N} | train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")
    print(f"Dims: P_obs={P_obs} | scaffold={cfg.scaffold} | U={U}")
    if mech_names is not None:
        print("Species:", ", ".join(str(x) for x in mech_names))

    best_val = float("inf")
    best_state = None
    epochs_since_improve = 0

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_species_losses: list[np.ndarray] = []

    def _save_ckpt(path: Path, epoch: int, tag: str):
        torch.save(
            {
                "epoch": int(epoch),
                "tag": str(tag),
                "state_dict": _clean_state_dict(model),
                "opt_state": opt.state_dict(),
                "best_val": float(best_val),
                "cfg": cfg.__dict__,
            },
            path,
        )

    if cfg.obs_idx is not None:
        obs_idx = torch.tensor(cfg.obs_idx, device=device, dtype=torch.long)
        print(f"Supervising only species indices: {cfg.obs_idx}")
    else:
        obs_idx = torch.arange(P_obs, device=device, dtype=torch.long)

    dt_tensor = torch.from_numpy(ds.dt).to(device)
    grouped_model = cfg.model_class == "ode_transformer_grouped"

    # ---- obs_normalization stats: fit on train split, applied in loss only ----
    obs_norm_method = str(cfg.obs_normalization)
    obs_norm_stats: Optional[dict] = None
    if obs_norm_method != "none":
        # Compute over valid timesteps of train split, then take obs_idx.
        train_idx_np = np.asarray(train_idx, dtype=np.int64)
        lens_np = ds.lengths if ds.variable_length else None
        parts = []
        for i in train_idx_np.tolist():
            L = int(lens_np[i]) if lens_np is not None else ds.y_seq.shape[1]
            parts.append(ds.y_seq[i, :L])
        all_y = np.concatenate(parts, axis=0)  # (N_steps_train, P_obs_full)
        # Slice to supervised obs_idx so shape matches pred/y_seq in loss_fn.
        obs_idx_np = obs_idx.detach().cpu().numpy()
        all_y_obs = all_y[:, obs_idx_np]
        m = all_y_obs.mean(axis=0).astype(np.float32)
        s = np.maximum(all_y_obs.std(axis=0), 1e-6).astype(np.float32)
        lo = all_y_obs.min(axis=0).astype(np.float32)
        hi = all_y_obs.max(axis=0).astype(np.float32)
        obs_norm_stats = {
            "mean": torch.from_numpy(m).to(device),
            "std":  torch.from_numpy(s).to(device),
            "min":  torch.from_numpy(lo).to(device),
            "max":  torch.from_numpy(hi).to(device),
        }
        print(f"obs_normalization: {obs_norm_method} | per-obs-species stats from "
              f"{len(train_idx_np)} train samples: mean={m.tolist()} std={s.tolist()} "
              f"min={lo.tolist()} max={hi.tolist()}")
        # Save for documentation / reproducibility. Distinct filename from old
        # train.py's `norm_stats.npz` (which transforms the dataset at LOAD time);
        # these stats apply at LOSS time only — predictions stay in raw scale.
        np.savez(
            exp_dir / "loss_norm_stats.npz",
            method=np.array(obs_norm_method, dtype="<U16"),
            obs_idx=obs_idx_np,
            mean=m, std=s, min=lo, max=hi,
        )

    # ---- species weights: computed in the SAME space the loss residuals live
    # in (obs_normalization, then log1p if loss_type=log_mse). Computing weights
    # on raw y while the loss is in normalized/log space silently mis-balances
    # species — e.g. with minmax both species are O(1) but raw-mean weights
    # leave pm ~16x weaker than mm.
    species_weights: Optional[torch.Tensor] = None
    if bool(cfg.normalize_by_species_mean):
        train_idx_np = np.asarray(train_idx, dtype=np.int64)
        lens_np = ds.lengths if ds.variable_length else None
        parts = []
        for i in train_idx_np.tolist():
            L = int(lens_np[i]) if lens_np is not None else ds.y_seq.shape[1]
            parts.append(ds.y_seq[i, :L])
        y_train = np.concatenate(parts, axis=0)  # (N_steps, P_obs_full)
        obs_idx_np = obs_idx.detach().cpu().numpy()
        y_obs = y_train[:, obs_idx_np].astype(np.float64)  # (N_steps, P_supervised)

        # Mirror _apply_obs_norm in numpy.
        if obs_norm_method == "none":
            y_loss = y_obs
        elif obs_norm_method == "sqrt":
            y_loss = np.sqrt(np.clip(y_obs, 0.0, None))
        elif obs_norm_method == "log1p":
            y_loss = np.log1p(np.clip(y_obs, 0.0, None))
        elif obs_norm_method == "log1p_clip1":
            y_loss = np.log1p(np.clip(y_obs, 1.0, None))
        elif obs_norm_method == "zscore":
            m = obs_norm_stats["mean"].detach().cpu().numpy().astype(np.float64)
            s = obs_norm_stats["std"].detach().cpu().numpy().astype(np.float64)
            y_loss = (y_obs - m) / s
        elif obs_norm_method == "minmax":
            lo = obs_norm_stats["min"].detach().cpu().numpy().astype(np.float64)
            hi = obs_norm_stats["max"].detach().cpu().numpy().astype(np.float64)
            y_loss = (y_obs - lo) / np.maximum(hi - lo, 1e-8)
        else:
            raise ValueError(f"Unknown obs_normalization: {obs_norm_method!r}")

        # If loss is log_mse, residuals further pass through log1p; mean must
        # too. (For zscore/minmax y_loss can be negative, but log_mse pairs
        # poorly with those anyway — flagged as a sanity-check combo.)
        if str(cfg.loss_type) == "log_mse":
            y_loss = np.log1p(np.clip(y_loss, 0.0, None))

        # Use mean(|y_loss|) so weights are well-defined even for zscore where
        # y_loss is centered. For non-negative spaces this matches mean(y).
        abs_mean = np.maximum(np.mean(np.abs(y_loss), axis=0), 1e-6).astype(np.float32)
        w_obs = (1.0 / abs_mean).astype(np.float32)
        species_weights = torch.from_numpy(w_obs).to(device)
        print(
            f"Species mean weights (obs, in loss space "
            f"[obs_norm={obs_norm_method}, loss={cfg.loss_type}]): "
            f"{species_weights.detach().cpu().tolist()}"
        )

    r_terminal_enabled = float(cfg.lambda_r_terminal) > 0.0
    r_idx_full: int | None = None
    if r_terminal_enabled:
        state_names = getattr(scaffold, "state_names", None)
        if state_names is None or "R" not in list(state_names):
            raise ValueError(
                f"lambda_r_terminal>0 but scaffold {cfg.scaffold} has no 'R' state."
            )
        r_idx_full = list(state_names).index("R")
        print(f"R terminal reg: lambda={cfg.lambda_r_terminal}, frac={cfg.r_terminal_frac}, r_idx_full={r_idx_full}")

    o_terminal_enabled = float(cfg.lambda_o_terminal) > 0.0
    o_idx_full: int | None = None
    if o_terminal_enabled:
        state_names = getattr(scaffold, "state_names", None)
        if state_names is None or "O" not in list(state_names):
            raise ValueError(
                f"lambda_o_terminal>0 but scaffold {cfg.scaffold} has no 'O' state."
            )
        o_idx_full = list(state_names).index("O")
        print(f"O terminal reg: lambda={cfg.lambda_o_terminal}, frac={cfg.o_terminal_frac}, o_idx_full={o_idx_full}")

    endpoint_enabled = float(cfg.lambda_endpoint) > 0.0
    if endpoint_enabled:
        print(f"Endpoint loss: lambda={cfg.lambda_endpoint}, mrna_obs_idx={cfg.endpoint_mrna_obs_idx}, protein_obs_idx={cfg.endpoint_protein_obs_idx}")

    # Frozen calibration scales for each aux term, populated on the first
    # training batch when `auto_scale_aux_losses` is True.
    aux_scale: dict[str, float] = {}

    for ep in range(1, cfg.epochs + 1):
        ep_t0 = time.time()
        teacher_forcing = bool(cfg.teacher_forcing) and (ep < int(cfg.tf_drop_epoch))

        # ---- train
        model.train()
        tr_total = 0.0
        tr_batches = 0
        theta_std_first_batch: float | None = None
        theta_mean_first_batch: float | None = None

        for y0, u_seq, y_seq, batch_lengths in train_loader:
            K_batch = u_seq.shape[1]
            dt_seq = dt_tensor[:K_batch][None, :].expand(y0.shape[0], -1)

            y0 = y0.to(device)
            y_seq = y_seq.to(device)
            u_seq = u_seq.to(device)
            dt_seq = dt_seq.to(device)
            if batch_lengths is not None:
                batch_lengths = batch_lengths.to(device)

            opt.zero_grad(set_to_none=True)
            model_kwargs = {
                "teacher_forcing": teacher_forcing,
                "tf_every": int(cfg.tf_every),
                "u_transform": str(cfg.u_transform),
                "y_transform": str(cfg.y_transform),
            }
            if grouped_model and batch_lengths is not None:
                model_kwargs["lengths"] = batch_lengths
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.autocast_bf16 and device.type == "cuda")):
                pred, theta, _ = model(
                    y0,
                    u_seq,
                    dt_seq,
                    obs_idx,
                    y_seq,
                    **model_kwargs,
                )
            pred_full = pred
            pred = pred[:, :, obs_idx]
            y_seq = y_seq[:, :, obs_idx]

            if theta_std_first_batch is None and theta is not None:
                with torch.no_grad():
                    th = theta.detach().float()
                    theta_std_first_batch = float(th.std(dim=0).mean().item())
                    theta_mean_first_batch = float(th.mean().item())

            loss = loss_fn(
                pred, y_seq, batch_lengths,
                loss_type=cfg.loss_type,
                species_weights=species_weights,
                obs_normalization=obs_norm_method,
                obs_norm_stats=obs_norm_stats,
            )
            # When `auto_scale_aux_losses` is True, each aux term is multiplied
            # by a FROZEN scaling factor `aux_scale[name]`, calibrated on the
            # first batch of training as `main_loss / aux_loss`. After that the
            # scale is constant, so λ has a stable cross-config meaning ("aux
            # contributes λ × main_loss" at the START of training) without the
            # positive-feedback loop where shrinking aux blows up the rescale.
            _AUX_EPS = 1e-12
            main_for_scale = loss.detach()
            relative = bool(getattr(cfg, "auto_scale_aux_losses", False))

            def _scaled_aux(aux, name):
                if not relative:
                    return aux
                if name not in aux_scale:
                    aux_scale[name] = float(
                        (main_for_scale / aux.detach().clamp(min=_AUX_EPS)).item()
                    )
                    print(f"[auto_scale] {name}: fixed scale = {aux_scale[name]:.4g} "
                          f"(main={float(main_for_scale.item()):.4g}, "
                          f"aux={float(aux.detach().item()):.4g})")
                return aux * aux_scale[name]

            if r_terminal_enabled:
                r_loss = r_terminal_loss(
                    pred_full, r_idx_full, batch_lengths, float(cfg.r_terminal_frac)
                )
                loss = loss + float(cfg.lambda_r_terminal) * _scaled_aux(r_loss, "r_terminal")

            if o_terminal_enabled:
                o_loss = r_terminal_loss(
                    pred_full, o_idx_full, batch_lengths, float(cfg.o_terminal_frac)
                )
                loss = loss + float(cfg.lambda_o_terminal) * _scaled_aux(o_loss, "o_terminal")

            if endpoint_enabled:
                ep_loss = endpoint_loss(
                    pred, y_seq, batch_lengths,
                    int(cfg.endpoint_mrna_obs_idx),
                    int(cfg.endpoint_protein_obs_idx),
                    species_weights=species_weights,
                )
                loss = loss + float(cfg.lambda_endpoint) * _scaled_aux(ep_loss, "endpoint")

            if float(cfg.lambda_pm_tail) > 0.0:
                t_loss = pm_tail_loss(
                    pred, y_seq, batch_lengths,
                    int(cfg.pm_tail_obs_idx),
                    float(cfg.pm_tail_frac),
                    obs_normalization=obs_norm_method,
                    obs_norm_stats=obs_norm_stats,
                )
                loss = loss + float(cfg.lambda_pm_tail) * _scaled_aux(t_loss, "pm_tail")

            if cfg.l1_regularization:
                reg_loss = torch.mean(torch.abs(theta[:,1:,:] - theta[:,:-1,:]))
                loss += cfg.lambda_reg * reg_loss

            if cfg.l2_regularization:
                reg_loss = torch.mean((theta[:,1:,:] - theta[:,:-1,:]).pow(2))
                loss += cfg.lambda_reg * reg_loss

            loss.backward()

            if cfg.grad_clip and float(cfg.grad_clip) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))

            opt.step()

            tr_total += float(loss.item())
            tr_batches += 1

        tr_loss = tr_total / max(1, tr_batches)
        train_losses.append(tr_loss)

        if scheduler is not None:
            scheduler.step()

        # ---- val
        va_loss = None
        sp_last = None

        if val_loader is not None:
            model.eval()
            va_total = 0.0
            va_batches = 0
            sp_total = None
            r_start_sum = 0.0
            r_mid_sum = 0.0
            r_end_sum = 0.0
            r_min_sum = 0.0
            r_tail_sum = 0.0
            r_count = 0

            with torch.no_grad():
                for y0, u_seq, y_seq, batch_lengths in val_loader:
                    K_batch = u_seq.shape[1]
                    dt_seq = torch.from_numpy(ds.dt[:K_batch])
                    dt_seq = dt_seq[None, :].expand(y0.shape[0], -1)

                    y0 = y0.to(device)
                    y_seq = y_seq.to(device)
                    u_seq = u_seq.to(device)
                    dt_seq = dt_seq.to(device)
                    if batch_lengths is not None:
                        batch_lengths = batch_lengths.to(device)

                    model_kwargs = {
                        "y_seq": None,
                        "teacher_forcing": False,
                        "u_transform": str(cfg.u_transform),
                        "y_transform": str(cfg.y_transform),
                    }
                    if grouped_model and batch_lengths is not None:
                        model_kwargs["lengths"] = batch_lengths
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.autocast_bf16 and device.type == "cuda")):
                        pred, _, _ = model(y0, u_seq, dt_seq, obs_idx, **model_kwargs)

                    if r_terminal_enabled and r_idx_full is not None:
                        Bv, Kv = pred.shape[0], pred.shape[1]
                        r_traj = pred[:, :, r_idx_full].float()  # (B, K)
                        if batch_lengths is None:
                            lens_v = torch.full((Bv,), Kv, device=pred.device, dtype=torch.long)
                        else:
                            lens_v = batch_lengths.long()
                        idx_end = (lens_v - 1).clamp(min=0)
                        idx_mid = (lens_v // 2).clamp(min=0)
                        arange_b = torch.arange(Bv, device=pred.device)
                        r_start_sum += float(r_traj[:, 0].sum().item())
                        r_mid_sum   += float(r_traj[arange_b, idx_mid].sum().item())
                        r_end_sum   += float(r_traj[arange_b, idx_end].sum().item())
                        valid_mask = (torch.arange(Kv, device=pred.device)[None, :] < lens_v[:, None])
                        r_masked = torch.where(valid_mask, r_traj, torch.full_like(r_traj, float("inf")))
                        r_min_sum += float(r_masked.min(dim=1).values.sum().item())
                        tail_mask = _build_tail_mask(
                            lens_v, Kv, pred.device, float(cfg.r_terminal_frac), Bv
                        ).to(r_traj.dtype)
                        tail_denom = tail_mask.sum(dim=1).clamp(min=1.0)
                        r_tail_sum += float(((r_traj * tail_mask).sum(dim=1) / tail_denom).sum().item())
                        r_count += Bv

                    pred = pred[:, :, obs_idx]
                    y_seq = y_seq[:, :, obs_idx]

                    loss = loss_fn(
                        pred, y_seq, batch_lengths,
                        loss_type=cfg.loss_type,
                        species_weights=species_weights,
                        obs_normalization=obs_norm_method,
                        obs_norm_stats=obs_norm_stats,
                    )
                    va_total += float(loss.item())

                    sp = loss_fn_per_species(
                        pred, y_seq, batch_lengths,
                        loss_type=cfg.loss_type,
                        species_weights=species_weights,
                        obs_normalization=obs_norm_method,
                        obs_norm_stats=obs_norm_stats,
                    ).detach().cpu()
                    sp_total = sp if sp_total is None else sp_total + sp
                    va_batches += 1

            va_loss = va_total / max(1, va_batches)
            val_losses.append(va_loss)

            r_stats = None
            if r_count > 0:
                r_stats = {
                    "start": r_start_sum / r_count,
                    "mid":   r_mid_sum   / r_count,
                    "end":   r_end_sum   / r_count,
                    "min":   r_min_sum   / r_count,
                    "tail":  r_tail_sum  / r_count,
                }

            if sp_total is not None:
                sp_last = (sp_total / max(1, va_batches)).numpy()
                val_species_losses.append(sp_last)

            if va_loss < best_val - float(cfg.early_stop_min_delta):
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

        ep_time = time.time() - ep_t0

        if va_loss is None:
            th_str = ""
            if theta_std_first_batch is not None:
                th_str = f"  theta[std_b={theta_std_first_batch:.3e} mean={theta_mean_first_batch:.3e}]"
            print(f"ep {ep:4d} | train {tr_loss:.6f} | tf={int(teacher_forcing)}{th_str} | {ep_time:.2f}s")
        else:
            sp_str = ""
            if sp_last is not None:
                if mech_names is None:
                    sp_str = "  [" + "  ".join(f"{v:.4f}" for v in sp_last) + "]"
                else:
                    sp_str = "  [" + "  ".join(f"{n}:{v:.4f}" for n, v in zip(mech_names, sp_last)) + "]"
            r_str = ""
            if r_stats is not None:
                r_str = (f"  R[t0={r_stats['start']:.3f} mid={r_stats['mid']:.3f} "
                         f"end={r_stats['end']:.3f} min={r_stats['min']:.3f} "
                         f"tail={r_stats['tail']:.3f}]")
            th_str = ""
            if theta_std_first_batch is not None:
                th_str = f"  theta[std_b={theta_std_first_batch:.3e} mean={theta_mean_first_batch:.3e}]"
            print(
                f"ep {ep:4d} | train {tr_loss:.6f} | val {va_loss:.6f} | best {best_val:.6f} | tf={int(teacher_forcing)}{sp_str}{r_str}{th_str} | {ep_time:.2f}s"
            )

        if math.isnan(tr_loss) or math.isnan(va_loss):
            print(f"NaN detected at epoch {ep} — stopping early.")
            break

        if int(cfg.early_stop_patience) > 0 and epochs_since_improve >= int(cfg.early_stop_patience):
            print(
                f"Early stop at epoch {ep}: no val improvement for {epochs_since_improve} epochs "
                f"(patience={int(cfg.early_stop_patience)}, best_val={best_val:.6f})."
            )
            break

        if wandb_run is not None:
            payload = {
                "epoch": int(ep),
                "train/loss": float(tr_loss),
                "train/teacher_forcing": int(teacher_forcing),
                "system/epoch_time_sec": float(ep_time),
                "system/learning_rate": float(opt.param_groups[0]["lr"]),
            }
            if va_loss is not None:
                payload["val/loss"] = float(va_loss)
                payload["val/best_loss"] = float(best_val)
            if sp_last is not None:
                names = mech_names if mech_names is not None else [f"species_{i}" for i in range(len(sp_last))]
                for name, value in zip(names, sp_last):
                    payload[f"val_species/{name}"] = float(value)
            if r_stats is not None:
                for k, v in r_stats.items():
                    payload[f"val_R/{k}"] = float(v)
            wandb_run.log(payload, step=int(ep))

        # always keep "last" checkpoint
        _save_ckpt(exp_dir / cfg.save_last_name, ep, tag="last")

        # periodic checkpoints for epoch-evolution overlays
        if int(cfg.ckpt_every) > 0 and (ep % int(cfg.ckpt_every) == 0):
            _save_ckpt(ckpt_dir / f"ckpt_ep{ep:04d}.pt", ep, tag="periodic")

        # write curves every epoch (so final plotting can use full history)
        curves_path = logs_dir / cfg.save_curves_name
        np.savez(
            curves_path,
            train_losses=np.array(train_losses, dtype=np.float32),
            val_losses=np.array(val_losses, dtype=np.float32) if len(val_losses) > 0 else None,
            val_species_losses=np.array(val_species_losses, dtype=np.float32) if len(val_species_losses) > 0 else None,
        )

    # restore best weights (if we had validation)
    if best_state is not None:
        model.load_state_dict(best_state)

    # final test evaluation (on held-out test set, using best weights)
    test_loss: float | None = None
    test_species_loss: np.ndarray | None = None
    if len(test_idx) > 0:
        test_loader = DataLoader(
            torch.utils.data.Subset(ds, test_idx.tolist()),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=bool(cfg.pin_memory),
        )
        model.eval()
        te_total = 0.0
        te_batches = 0
        sp_total = None
        with torch.no_grad():
            for y0, u_seq, y_seq, batch_lengths in test_loader:
                K_batch = u_seq.shape[1]
                dt_seq = dt_tensor[:K_batch][None, :].expand(y0.shape[0], -1)
                y0 = y0.to(device)
                y_seq = y_seq.to(device)
                u_seq = u_seq.to(device)
                dt_seq = dt_seq.to(device)
                if batch_lengths is not None:
                    batch_lengths = batch_lengths.to(device)
                model_kwargs = {
                    "y_seq": None,
                    "teacher_forcing": False,
                    "u_transform": str(cfg.u_transform),
                    "y_transform": str(cfg.y_transform),
                }
                if grouped_model and batch_lengths is not None:
                    model_kwargs["lengths"] = batch_lengths
                pred, _, _ = model(y0, u_seq, dt_seq, obs_idx, **model_kwargs)
                pred = pred[:, :, obs_idx]
                y_seq = y_seq[:, :, obs_idx]
                loss = loss_fn(
                    pred, y_seq, batch_lengths,
                    loss_type=cfg.loss_type,
                    species_weights=species_weights,
                    obs_normalization=obs_norm_method,
                    obs_norm_stats=obs_norm_stats,
                )
                te_total += float(loss.item())
                sp = loss_fn_per_species(
                    pred, y_seq, batch_lengths,
                    loss_type=cfg.loss_type,
                    species_weights=species_weights,
                    obs_normalization=obs_norm_method,
                    obs_norm_stats=obs_norm_stats,
                ).detach().cpu()
                sp_total = sp if sp_total is None else sp_total + sp
                te_batches += 1
        test_loss = te_total / max(1, te_batches)
        if sp_total is not None:
            test_species_loss = (sp_total / max(1, te_batches)).numpy()
            sp_str = "  [" + "  ".join(
                f"{n}:{v:.4f}" for n, v in zip(
                    mech_names if mech_names else [f"s{i}" for i in range(len(test_species_loss))],
                    test_species_loss,
                )
            ) + "]"
        else:
            sp_str = ""
        print(f"\nTest loss (best model): {test_loss:.6f}{sp_str}")

    # write final loss_curves.npz including test results
    np.savez(
        logs_dir / cfg.save_curves_name,
        train_losses=np.array(train_losses, dtype=np.float32),
        val_losses=np.array(val_losses, dtype=np.float32) if len(val_losses) > 0 else None,
        val_species_losses=np.array(val_species_losses, dtype=np.float32) if len(val_species_losses) > 0 else None,
        test_loss=np.float32(test_loss) if test_loss is not None else None,
        test_species_losses=test_species_loss.astype(np.float32) if test_species_loss is not None else None,
    )

    # save best model (plot expects exp_dir/model.pt)
    save_path = exp_dir / cfg.save_model_name
    torch.save(
        {"state_dict": _clean_state_dict(model),
         "best_val": float(best_val),
         "test_loss": float(test_loss) if test_loss is not None else None},
        save_path,
    )
    print(f"Saved best model to {save_path}")

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.2f}s ({elapsed/60:.2f}m)")
    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    # plots ONLY at the end (including epoch evolution overlays from checkpoints)
    if not no_plot:
        try:
            from plot_diagnostics import plot_experiment, plot_epoch_prediction_overlays

            plot_experiment(exp_dir, n_samples=int(plot_samples), sample_idx=int(plot_sample_idx))

            # epochs=None => automatically uses available checkpoints and picks up to max_overlays evenly spaced
            plot_epoch_prediction_overlays(
                exp_dir,
                sample_idx=int(plot_sample_idx),
                epochs=None,
                max_overlays=8,
            )
        except ImportError:
            print("[plot] plot_diagnostics.py not found; skipping plots.")
        except Exception as e:
            print(f"[plot] failed: {e}")

    if wandb_run is not None:
        plots_dir = exp_dir / "plots"
        log_wandb_images(wandb, wandb_run, plots_dir)
        log_wandb_artifact(wandb, wandb_run, exp_dir=exp_dir, run_id=run_id)
        wandb_run.summary["run_dir"] = str(exp_dir.resolve())
        wandb_run.summary["study"] = cfg.study
        wandb_run.summary["scaffold"] = cfg.scaffold
        wandb_run.summary["device"] = str(device)
        wandb_run.summary["elapsed_seconds"] = float(elapsed)
        if train_losses:
            wandb_run.summary["final_train_loss"] = float(train_losses[-1])
        if val_losses:
            wandb_run.summary["final_val_loss"] = float(val_losses[-1])
        if best_state is not None:
            wandb_run.summary["best_val_loss"] = float(best_val)
        wandb_run.finish()

    # Endpoint R^2 analysis (optional)
    if cfg.endpoint_r2:
        try:
            from metrics import endpoint_r2
            from plot_diagnostics import device_auto as _dev_auto

            r2_device = _dev_auto()
            print("\nRunning endpoint R^2 analysis...")
            result = endpoint_r2.collect_endpoints(
                exp_dir, r2_device, split="test", protein_sp="pm", mrna_sp="mm"
            )
            r2_protein = endpoint_r2.r2(result["true_protein_final"], result["pred_protein_final"])
            r2_mrna = endpoint_r2.r2(result["true_mrna_max"], result["pred_mrna_max"])
            print(f"  R²(protein final) = {r2_protein:.4f}")
            print(f"  R²(mRNA max)      = {r2_mrna:.4f}")

            out_path = exp_dir / "endpoint_r2.png"
            endpoint_r2.plot_endpoints([result], protein_sp="pm", mrna_sp="mm",
                                       split="test", out_path=out_path)

            if wandb_run is not None:
                wandb_run.summary["endpoint_r2/protein_final"] = float(r2_protein)
                wandb_run.summary["endpoint_r2/mrna_max"] = float(r2_mrna)
        except Exception as e:
            print(f"[endpoint_r2] failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-samples", type=int, default=5)
    parser.add_argument("--plot-sample-idx", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override any config field, e.g. --set lr=1e-3 --set scaffold=mof_synthesis_8")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    for kv in args.set:
        key, val = kv.split("=", 1)
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config field: {key!r}. Valid fields: {list(vars(cfg).keys())}")
        parsed = yaml.safe_load(val)
        # yaml.safe_load("None") → str "None", not Python None — fix explicitly
        if isinstance(parsed, str) and parsed == "None":
            parsed = None
        orig_type = type(getattr(cfg, key))
        if parsed is not None and orig_type is not type(None) and not isinstance(parsed, orig_type):
            parsed = orig_type(parsed)
        setattr(cfg, key, parsed)

    train(cfg, no_plot=bool(args.no_plot), plot_samples=int(args.plot_samples), plot_sample_idx=int(args.plot_sample_idx))
