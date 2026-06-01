# train.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional
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

    def __init__(self, npz_path: str | Path, *, use_synthetic_data: bool = True,
                 synth_max_samples: int | None = None, seed: int = 0):
        d = np.load(str(npz_path), allow_pickle=True)
        # Optional ablation: drop synthetic no-go rows (z_expr == 0) before any
        # other field is processed. Useful for A/B-testing whether the synth
        # bulk helps or hurts (set via cfg.use_synthetic_data).
        _filter_mask: np.ndarray | None = None
        if not use_synthetic_data and "z_expr" in d.files:
            _filter_mask = (d["z_expr"].astype(np.int64) == 1)
            n_kept = int(_filter_mask.sum())
            n_drop = int((~_filter_mask).sum())
            print(f"ODEDataset: use_synthetic_data=False → dropping {n_drop} synth rows, keeping {n_kept} real.")
        # Optional hard cap on synth rows. Subsample synth to at most
        # synth_max_samples, keeping all real rows intact. This gives the same
        # regularisation effect as a very low source_loss_weight but at real-data
        # compute cost (no wasted forward/backward on discarded synth samples).
        elif synth_max_samples is not None and "z_expr" in d.files:
            z = d["z_expr"].astype(np.int64)
            real_idx = np.where(z == 1)[0]
            synth_idx = np.where(z == 0)[0]
            n_synth = len(synth_idx)
            if n_synth > synth_max_samples:
                rng = np.random.default_rng(seed)
                synth_keep = rng.choice(synth_idx, size=synth_max_samples, replace=False)
                synth_keep.sort()
                keep = np.concatenate([real_idx, synth_keep])
                keep.sort()
                _filter_mask = np.zeros(len(z), dtype=bool)
                _filter_mask[keep] = True
                print(f"ODEDataset: synth_max_samples={synth_max_samples} → "
                      f"subsampled {n_synth} → {synth_max_samples} synth rows, "
                      f"keeping all {len(real_idx)} real rows.")
        def _maybe_filter(arr: np.ndarray) -> np.ndarray:
            return arr[_filter_mask] if _filter_mask is not None else arr

        self.y0 = _maybe_filter(d["y0"].astype(np.float32))                # (N,P_obs)
        self.u_seq = _maybe_filter(d["u_seq"].astype(np.float32))          # (N,K,U)
        self.y_seq = _maybe_filter(d["y_seq"].astype(np.float32))          # (N,K,P_obs)
        t_obs = d["t_obs"].astype(np.float32)  # (K+1,)
        self.dt = np.diff(t_obs).astype(np.float32)  # (K,) — shared fallback when no per-sample grid

        # Per-sample time grid (optional). When present, lets datasets mix
        # samples from different acquisition grids (e.g. OLD 60s + NEW 160s)
        # without resampling. Shape (N, K_max). Padded steps for shorter
        # samples can be any value — they're masked out via `lengths`.
        if "dt_per_sample" in d.files:
            self.dt_per_sample = _maybe_filter(d["dt_per_sample"].astype(np.float32))
        else:
            self.dt_per_sample = None

        if "control_indices" not in d or "obs_indices" not in d:
            raise ValueError(
                f"Dataset {npz_path} missing control_indices/obs_indices. Regenerate dataset with metadata."
            )
        self.control_indices = d["control_indices"].astype(np.int64)
        self.obs_indices = d["obs_indices"].astype(np.int64)

        self.names_full = d["names_full"].astype(str) if "names_full" in d else None
        self.control_names = d["control_names"].astype(str) if "control_names" in d else None
        self.obs_names = d["obs_names"].astype(str) if "obs_names" in d else None
        # Optional mapping from this (possibly filtered) dataset back to the
        # original row indices in the source dataset. When present, callers
        # can remap hardcoded splits that refer to the original indexing.
        self.original_indices = (_maybe_filter(d["original_indices"].astype(np.int64))
                                  if "original_indices" in d else None)

        # MinMax stats for u (per channel). Mirrors train_R.py so replot works
        # with models trained via either training script.
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
            self.lengths = _maybe_filter(d["lengths"].astype(np.int64))    # (N,)
            self.variable_length = True
        else:
            self.lengths = None
            self.variable_length = False

        # Optional Model 7 boundary label. 1 = real expression run, 0 = synthetic
        # no-go. Written by scripts/build_txtl_combined_npz.py. Absent on legacy
        # datasets, in which case the zero-trajectory loss is a no-op.
        # After the optional filter above, this is either all-ones or absent.
        self.z_expr = (_maybe_filter(d["z_expr"].astype(np.int64))
                       if "z_expr" in d else None)

        # Per-sample source-domain index: 0=old, 1=new, 2=synth, -1=unknown.
        # Plumbed through __getitem__ → collate → train loop so the loss can
        # upweight a domain (source_loss_weights) or the sampler can balance
        # batches across domains (balance_source_sampler). Absent on legacy
        # datasets where the source isn't recorded.
        if "source_label" in d.files:
            _src_to_int = {"old": 0, "new": 1, "synth": 2}
            _src_arr = np.array([_src_to_int.get(str(s), -1) for s in d["source_label"]], dtype=np.int64)
            self.source_idx = _maybe_filter(_src_arr)
        else:
            self.source_idx = None

    def __len__(self) -> int:
        return self.y0.shape[0]

    def __getitem__(self, i: int):
        # z_expr=1 default (real) when the dataset doesn't carry the label, so
        # the boundary loss is a no-op on legacy datasets.
        z_i = int(self.z_expr[i]) if self.z_expr is not None else 1
        # source_idx: 0=old, 1=new, 2=synth, -1 (unknown) when dataset has no source_label.
        s_i = int(self.source_idx[i]) if self.source_idx is not None else -1
        if self.variable_length:
            L = int(self.lengths[i])
            dt_i = (self.dt_per_sample[i, :L] if self.dt_per_sample is not None
                    else self.dt[:L])
            return (
                torch.from_numpy(self.y0[i]),          # (P_obs,)
                torch.from_numpy(self.u_seq[i, :L]),   # (L,U)
                torch.from_numpy(self.y_seq[i, :L]),   # (L,P_obs)
                torch.from_numpy(np.ascontiguousarray(dt_i)),  # (L,)
                torch.tensor(z_i, dtype=torch.long),
                torch.tensor(s_i, dtype=torch.long),
            )
        dt_i = (self.dt_per_sample[i] if self.dt_per_sample is not None
                else self.dt)
        return (
            torch.from_numpy(self.y0[i]),  # (P_obs,)
            torch.from_numpy(self.u_seq[i]),  # (K,U)
            torch.from_numpy(self.y_seq[i]),  # (K,P_obs)
            torch.from_numpy(np.ascontiguousarray(dt_i)),  # (K,)
            torch.tensor(z_i, dtype=torch.long),
            torch.tensor(s_i, dtype=torch.long),
        )


def collate(batch):
    y0, u, y, dt, z, s = zip(*batch)
    return (torch.stack(y0), torch.stack(u), torch.stack(y), None,
            torch.stack(z), torch.stack(dt), torch.stack(s))


def collate_varlen(batch):
    """Pad each batch to its own max length; return lengths tensor."""
    y0_list, u_list, y_list, dt_list, z_list, s_list = zip(*batch)
    lengths = torch.tensor([u.shape[0] for u in u_list], dtype=torch.long)
    y0 = torch.stack(y0_list)
    u_padded = torch.nn.utils.rnn.pad_sequence(u_list, batch_first=True)   # (B, K_batch, U)
    y_padded = torch.nn.utils.rnn.pad_sequence(y_list, batch_first=True)   # (B, K_batch, P)
    dt_padded = torch.nn.utils.rnn.pad_sequence(dt_list, batch_first=True)  # (B, K_batch) — padded with 0
    z = torch.stack(z_list)
    s = torch.stack(s_list)
    return y0, u_padded, y_padded, lengths, z, dt_padded, s


def _build_loss_mask(lengths: torch.Tensor, K: int, device: torch.device) -> torch.Tensor:
    """Build (B, K) boolean mask: True for valid timesteps."""
    return torch.arange(K, device=device).unsqueeze(0) < lengths.unsqueeze(1)


# Resource-state IC names that vary between dataset and scaffold conventions.
# Used by _lift_to_scaffold_state to copy initial conditions even when the
# scaffold renames a state (e.g. dataset "O" vs M9's "O2"). Add entries here
# if a new scaffold uses yet another naming.
_LIFT_NAME_ALIASES: dict[str, str] = {
    "O2": "O",   # M9 oxygen state is called "O2"; dataset stores it as "O"
}


def _lift_to_scaffold_state(
    y0: torch.Tensor,
    y_seq: torch.Tensor,
    dataset_obs_idx: list[int],
    scaffold_obs_idx: list[int],
    scaffold_P: int,
    dataset_state_names: list[str] | None = None,
    scaffold_state_names: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-pack dataset observations into a scaffold-shape state vector.

    The dataset's y0 / y_seq are (B, P_obs) and (B, K, P_obs) in dataset layout
    (e.g. txtl_combined.npz is laid out like the 7-state Model 5; mm/pm sit at
    dataset_obs_idx = [3, 5]; the other cols are placeholder zeros).

    For a partially-observed scaffold (different P, mm/pm at different cols), we:
      1. Extract the observed channels and place them at scaffold_obs_idx in a
         zero-filled (B, scaffold_P) tensor (this carries the measured mm/pm
         signal through to the supervised positions).
      2. ALSO copy any *non-observed* dataset state whose name matches a
         scaffold state name (with the _LIFT_NAME_ALIASES table). This carries
         resource/cofactor initial conditions (R=1, O=1 set by the data builder)
         into y0. Without this, scaffolds whose dynamics multiply by R (M5/M7/M9)
         silently get R(0)=0 → R(t)=0 → entire transcription cascade dies.

    y_seq columns for non-observed states stay zero — the loss only supervises
    the obs_state_idx columns, and the model integrates its own trajectory
    forward from y0, so y_seq[..., non_obs] is never read.
    """
    if len(dataset_obs_idx) != len(scaffold_obs_idx):
        raise ValueError(
            f"dataset_obs_idx (len={len(dataset_obs_idx)}) must match "
            f"scaffold_obs_idx (len={len(scaffold_obs_idx)})."
        )
    B = y0.shape[0]
    K = y_seq.shape[1]
    src = torch.as_tensor(dataset_obs_idx, device=y0.device, dtype=torch.long)
    dst = torch.as_tensor(scaffold_obs_idx, device=y0.device, dtype=torch.long)

    y0_obs    = y0.index_select(1, src)                 # (B, n_obs)
    y_seq_obs = y_seq.index_select(2, src)              # (B, K, n_obs)

    y0_full    = torch.zeros((B, scaffold_P),    device=y0.device,    dtype=y0.dtype)
    y_seq_full = torch.zeros((B, K, scaffold_P), device=y_seq.device, dtype=y_seq.dtype)
    y0_full.index_copy_(1, dst, y0_obs)
    y_seq_full.index_copy_(2, dst, y_seq_obs)

    # Step 2: by-name IC copy for non-observed states (resource/cofactor).
    # Only fires when both name lists are provided (kept optional for back-compat).
    if dataset_state_names is not None and scaffold_state_names is not None:
        ds_name_to_idx = {str(n): i for i, n in enumerate(dataset_state_names)}
        scaf_obs_set = set(int(i) for i in scaffold_obs_idx)
        for s_idx, s_name in enumerate(scaffold_state_names):
            if s_idx in scaf_obs_set:
                continue  # already populated in step 1
            ds_name = _LIFT_NAME_ALIASES.get(str(s_name), str(s_name))
            if ds_name in ds_name_to_idx:
                d_idx = ds_name_to_idx[ds_name]
                y0_full[:, s_idx] = y0[:, d_idx]
    return y0_full, y_seq_full


def _apply_channel_min_gate(
    y0: torch.Tensor,
    y_seq: torch.Tensor,
    cols: list[int] | None,
    lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample, per-channel min-subtraction on (y0, y_seq) for a batch.

    Lifts the lowest observed value of each (experiment, channel) to ~0 so
    failure/baseline traces sit near zero ("gate the observation"). This is a
    pure runtime op on the batch; the underlying npz is never modified.

    cols=None applies to every channel; otherwise restrict to the listed indices.

    If `lengths` is provided, padded positions (k >= length) are masked out of
    the min so a zero-padded NPZ doesn't trivially set ch_min=0 and turn the
    gate into a no-op on shorter samples.
    """
    B, K, P = y_seq.shape
    if lengths is not None:
        ar = torch.arange(K, device=y_seq.device).unsqueeze(0)        # (1, K)
        valid = ar < lengths.to(y_seq.device).unsqueeze(1)            # (B, K)
        mask = valid.unsqueeze(-1)                                    # (B, K, 1)
        masked_y = torch.where(mask, y_seq, torch.full_like(y_seq, float("inf")))
    else:
        masked_y = y_seq
    if cols is None:
        ch_min = masked_y.amin(dim=1, keepdim=True)                   # (B,1,P)
        return y0 - ch_min[:, 0, :], y_seq - ch_min
    idx = torch.as_tensor(cols, device=y_seq.device, dtype=torch.long)
    ch_min = masked_y.index_select(dim=2, index=idx).amin(dim=1, keepdim=True)  # (B,1,|cols|)
    y0_out = y0.clone()
    y_seq_out = y_seq.clone()
    y0_out[:, idx] = y0[:, idx] - ch_min[:, 0, :]
    y_seq_out[:, :, idx] = y_seq[:, :, idx] - ch_min
    return y0_out, y_seq_out


def _load_fixed_split_triples(
    path: str | Path,
    idx: int,
    original_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the (train, val, test) triple at position `idx` from an `.npz` file.

    Expected keys: ``train_{i}``, ``val_{i}``, ``test_{i}`` for i in [0, n_triples).
    If `original_indices` is provided (filtered dataset), the stored indices are
    remapped to their new positions and any indices that were filtered out are
    dropped.
    """
    with np.load(path) as f:
        keys = set(f.files)
        n = int(f["n_triples"]) if "n_triples" in keys else \
            sum(1 for k in keys if k.startswith("train_"))
        if not (0 <= idx < n):
            raise ValueError(f"fixed_split_idx={idx} out of range; file has {n} triples")
        train_raw = f[f"train_{idx}"].astype(np.int64)
        val_raw   = f[f"val_{idx}"].astype(np.int64)
        test_raw  = f[f"test_{idx}"].astype(np.int64)

    if original_indices is None:
        return train_raw, val_raw, test_raw

    mapping = {int(orig): new_i for new_i, orig in enumerate(original_indices)}
    keep = lambda arr: np.asarray([mapping[int(i)] for i in arr if int(i) in mapping],
                                  dtype=np.int64)
    return keep(train_raw), keep(val_raw), keep(test_raw)


def loss_fn(
    pred: torch.Tensor,
    y_seq: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    use_log_loss: bool = True,
    channel_weights: Optional[torch.Tensor] = None,
    time_weight: Optional[torch.Tensor] = None,
    clamp_min: float = 0.0,
    sample_weights: Optional[torch.Tensor] = None,
    sample_channel_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE loss, optionally in log1p space. use_log_loss=False when data is pre-normalised.

    Base path (channel_weights=None, time_weight=None) is unchanged: masked MSE
    averaged over (valid timesteps × channels) with a shared denominator.

    Composable extensions used by the supervisor-style loss:
      * channel_weights: (P,) per-species multipliers. When given, channels are
        normalised independently and combined as Σ_p w_p · mse_p (no /P division),
        so relative scales between species are preserved.
      * time_weight: (B, K, P) per-(sample, timestep, species) weight. When given
        it replaces the lengths-mask entirely (must already encode validity).
      * sample_channel_mask: (B, P) float tensor — 0 suppresses a channel for a
        specific sample. Used to zero out pm on synth no-expression samples so
        those trajectories (pm≈0) don't bias the model toward predicting zero
        protein on real expressing samples.
    """
    if use_log_loss:
        if clamp_min > 0.0:
            # Supervisor parity: floor both pred and y at clamp_min before log1p,
            # creating a dead zone in [0, clamp_min] where small-value mismatches
            # don't contribute to the loss.
            pred  = torch.log1p(pred.clamp_min(clamp_min))
            y_seq = torch.log1p(y_seq.clamp_min(clamp_min))
        else:
            pred  = torch.log1p(pred)
            y_seq = torch.log1p(y_seq)
    se = (pred - y_seq).pow(2)  # (B,K,P)

    if time_weight is None:
        if lengths is not None:
            mask = _build_loss_mask(lengths, se.shape[1], se.device)  # (B,K)
            w = mask.unsqueeze(-1).to(se.dtype).expand_as(se)
        else:
            w = torch.ones_like(se)
    else:
        w = time_weight.to(se.dtype)

    # Per-sample weights (e.g. domain upweighting for under-represented sources).
    # `expand_as` returned a view of the lengths mask, so .contiguous().clone()
    # before in-place multiplication to avoid silently scaling the source mask.
    if sample_weights is not None:
        w = w.contiguous().clone()
        sw = sample_weights.view(-1, 1, 1).to(se.dtype)
        w = w * sw

    # Per-sample per-channel mask: e.g. zero pm for synth no-expression samples
    # so they don't teach the model to predict zero protein on real experiments.
    # Shape (B, P) — broadcast over K via unsqueeze(1).
    if sample_channel_mask is not None:
        w = w.contiguous().clone()
        w = w * sample_channel_mask.unsqueeze(1).to(se.dtype)

    if channel_weights is None:
        return (se * w).sum() / w.sum().clamp_min(1.0)

    num = (se * w).sum(dim=(0, 1))            # (P,)
    den = w.sum(dim=(0, 1)).clamp_min(1.0)    # (P,)
    mse_per_chan = num / den
    return (channel_weights.to(mse_per_chan.dtype) * mse_per_chan).sum()


def _resolve_channel_weights(
    obs_idx_sliced: list[int],
    species_weights: dict | None,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Map raw-state-index weights to a (P,) tensor over the post-slice ordering.
    Missing species default to 1.0. Returns None when no weights are configured.
    """
    if not species_weights:
        return None
    w = [float(species_weights.get(int(s), species_weights.get(str(s), 1.0)))
         for s in obs_idx_sliced]
    return torch.tensor(w, device=device, dtype=dtype)


def _build_per_channel_time_weight(
    obs_idx_sliced: list[int],
    time_upweight: dict | None,
    lengths: Optional[torch.Tensor],
    B: int,
    K: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Build a (B, K, P) weight tensor combining validity mask + per-channel
    sub-interval upweighting.

    time_upweight maps {raw_state_idx: [start_frac, factor]}: for that channel,
    timesteps t >= int(start_frac · length) are multiplied by factor. Channels
    not listed get the plain validity mask.
    """
    if not time_upweight:
        return None
    if lengths is None:
        lengths_t = torch.full((B,), K, device=device, dtype=torch.long)
    else:
        lengths_t = lengths.to(device=device, dtype=torch.long)
    t_grid = torch.arange(K, device=device)
    valid = (t_grid[None, :] < lengths_t[:, None]).to(dtype)  # (B, K)
    P = len(obs_idx_sliced)
    w = valid.unsqueeze(-1).expand(B, K, P).clone()
    for p, raw in enumerate(obs_idx_sliced):
        spec = time_upweight.get(int(raw), time_upweight.get(str(raw)))
        if not spec:
            continue
        start_frac, factor = float(spec[0]), float(spec[1])
        start_idx = (lengths_t.to(torch.float32) * start_frac).to(torch.long)
        up = (t_grid[None, :] >= start_idx[:, None]) & (t_grid[None, :] < lengths_t[:, None])
        w[:, :, p] = torch.where(up, w[:, :, p] * factor, w[:, :, p])
    return w


def endpoint_mse(
    pred: torch.Tensor,
    y_seq: torch.Tensor,
    lengths: Optional[torch.Tensor],
    channels: list[int],
    use_log_loss: bool = True,
    clamp_min: float = 0.0,
    sample_mask: Optional[torch.Tensor] = None,
    loss_space: str = "log",
) -> torch.Tensor:
    """MSE at the final valid timestep on selected post-slice channels.
    `channels` indexes into the sliced (B, K, P) tensors, not raw state indices.
    `clamp_min`: floor pred/y before log1p (supervisor parity; 0 disables).
    `sample_mask`: (B,) bool — False rows are excluded from this loss term.
                   Used to skip synth samples when endpoint channels are masked.
    `loss_space`: comparison space for the endpoint term.
        "log"    — log1p space (DEFAULT; original behaviour, gated by use_log_loss).
        "sqrt"   — sqrt space: penalises high-yield misses ~proportionally to the
                   value, countering the log1p tail-underprediction bias without
                   the magnitude blow-up of raw linear MSE.
        "linear" — raw linear MSE (full high-end weight; can be unstable).
      Note: when loss_space != "log", `use_log_loss` is ignored for this term so
      the endpoint term can use a different (less-compressive) space than the
      trajectory loss.
    """
    if loss_space == "sqrt":
        # clamp_min floors values before sqrt (same dead-zone semantics as log path)
        cm = clamp_min if clamp_min > 0.0 else 0.0
        pred  = pred.clamp_min(cm).sqrt()
        y_seq = y_seq.clamp_min(cm).sqrt()
    elif loss_space == "linear":
        cm = clamp_min if clamp_min > 0.0 else 0.0
        pred  = pred.clamp_min(cm)
        y_seq = y_seq.clamp_min(cm)
    elif use_log_loss:  # loss_space == "log" (default)
        if clamp_min > 0.0:
            pred  = torch.log1p(pred.clamp_min(clamp_min))
            y_seq = torch.log1p(y_seq.clamp_min(clamp_min))
        else:
            pred  = torch.log1p(pred)
            y_seq = torch.log1p(y_seq)
    B, K, _ = pred.shape
    if lengths is None:
        last = torch.full((B,), K - 1, device=pred.device, dtype=torch.long)
    else:
        last = (lengths.to(pred.device, dtype=torch.long) - 1).clamp_min(0)
    batch_idx = torch.arange(B, device=pred.device)
    chans = torch.tensor(channels, device=pred.device, dtype=torch.long)
    p_last = pred[batch_idx[:, None], last[:, None], chans[None, :]]
    y_last = y_seq[batch_idx[:, None], last[:, None], chans[None, :]]
    if sample_mask is not None:
        keep = sample_mask.to(pred.device)
        p_last = p_last[keep]
        y_last = y_last[keep]
    if p_last.numel() == 0:
        return pred.new_zeros(())
    return (p_last - y_last).pow(2).mean()


def peak_mse(
    pred: torch.Tensor,
    y_seq: torch.Tensor,
    lengths: Optional[torch.Tensor],
    channels: list[int],
    use_log_loss: bool = True,
    clamp_min: float = 0.0,
    sample_mask: Optional[torch.Tensor] = None,
    loss_space: str = "log",
) -> torch.Tensor:
    """MSE on the per-trajectory MAX value of selected post-slice channels.

    The analog of endpoint_mse but for the trajectory PEAK rather than the final
    point. Motivated by the R²(mRNA-max) metric and the M4/M5/M9 cascade: mRNA
    rises then degrades, so its informative quantity is the interior peak, not
    the endpoint. Anchoring mRNA-max (channel 3) + protein-final (endpoint_mse,
    channel 5) pins both ends of the transcription→translation→maturation
    cascade; the ODE enforces consistency between them.

    `channels` indexes the sliced (B,K,P) tensors. `loss_space`: "log" (default),
    "sqrt", or "linear" — same semantics as endpoint_mse. Max is sub-
    differentiable (gradient flows to the argmax timestep), like maxpool.
    """
    # Mask padded timesteps to -inf so amax ignores them.
    B, K, _ = pred.shape
    if lengths is not None:
        ar = torch.arange(K, device=pred.device).unsqueeze(0)
        valid = ar < lengths.to(pred.device).unsqueeze(1)          # (B,K)
        neg_inf = torch.finfo(pred.dtype).min
        mask3 = valid.unsqueeze(-1)
    chans = torch.tensor(channels, device=pred.device, dtype=torch.long)
    p_sel = pred.index_select(dim=2, index=chans)                  # (B,K,|chans|)
    y_sel = y_seq.index_select(dim=2, index=chans)
    if lengths is not None:
        p_sel = torch.where(mask3, p_sel, torch.full_like(p_sel, neg_inf))
        y_sel = torch.where(mask3, y_sel, torch.full_like(y_sel, neg_inf))
    p_peak = p_sel.amax(dim=1)   # (B, |chans|)
    y_peak = y_sel.amax(dim=1)

    if loss_space == "sqrt":
        cm = clamp_min if clamp_min > 0.0 else 0.0
        p_peak = p_peak.clamp_min(cm).sqrt()
        y_peak = y_peak.clamp_min(cm).sqrt()
    elif loss_space == "linear":
        cm = clamp_min if clamp_min > 0.0 else 0.0
        p_peak = p_peak.clamp_min(cm)
        y_peak = y_peak.clamp_min(cm)
    elif use_log_loss:
        if clamp_min > 0.0:
            p_peak = torch.log1p(p_peak.clamp_min(clamp_min))
            y_peak = torch.log1p(y_peak.clamp_min(clamp_min))
        else:
            p_peak = torch.log1p(p_peak.clamp_min(0.0))
            y_peak = torch.log1p(y_peak.clamp_min(0.0))

    if sample_mask is not None:
        keep = sample_mask.to(pred.device)
        p_peak = p_peak[keep]
        y_peak = y_peak[keep]
    if p_peak.numel() == 0:
        return pred.new_zeros(())
    return (p_peak - y_peak).pow(2).mean()


# DEPRECATED: superseded by the composable knobs on loss_fn (channel_weights,
# time_weight) + endpoint_mse + cfg.loss_normalizer_channels. Kept temporarily
# for back-compat with `use_ivtt_mse_loss: true` configs and bit-exact parity
# checks against the supervisor's original code. Remove once new-path runs are
# confirmed to match.
#
# Note on exact parity vs the new path: this function applies
# log1p(clamp_min(y, 1.0)) (floors values at 1.0 before log1p), whereas the new
# composable path uses plain log1p(y). For values near zero this differs
# (log1p(0)=0 vs log1p(1)=log 2 ≈ 0.693). So even with species_weights /
# time_upweight / lambda_endpoint / loss_normalizer_channels all set to the
# supervisor's recipe, results will not be bit-exact. If exact reproduction is
# needed later, add a `loss_clamp_min` knob to the new path.
def loss_fn_ivtt_mse(
    *,
    pred_full: torch.Tensor,
    theta: torch.Tensor,
    y_seq_full: torch.Tensor,
    lengths: Optional[torch.Tensor],
    mm_state_idx: int = 3,
    pm_state_idx: int = 5,
) -> torch.Tensor:
    """Replicate bob_model/neural_spline.py:NeuralSpline.train_val() loss.

    Assumes last-layer-ode 7-state layout where mm is at index 3 and pm at 5.
    Uses Bob's fixed loss_weight vector and masking/weighting scheme.
    """
    B, K, _ = y_seq_full.shape
    device = y_seq_full.device
    dtype = y_seq_full.dtype

    if lengths is None:
        lengths = torch.full((B,), K, device=device, dtype=torch.long)

    y_mm = y_seq_full[:, :, mm_state_idx:mm_state_idx + 1]
    y_pm = y_seq_full[:, :, pm_state_idx:pm_state_idx + 1]
    p_zero = torch.zeros_like(y_mm)
    y_seq_3 = torch.cat([y_mm, p_zero, y_pm], dim=-1)  # (B,K,3)

    pred_mm = pred_full[:, :, mm_state_idx:mm_state_idx + 1]
    pred_pm = pred_full[:, :, pm_state_idx:pm_state_idx + 1]
    pred_3 = torch.cat([pred_mm, torch.zeros_like(pred_mm), pred_pm], dim=-1)  # (B,K,3)

    y_clamped = y_seq_3.clamp_min(1.0)
    pred_clamped = pred_3.clamp_min(1.0)
    log_y = torch.log1p(y_clamped)
    log_pred = torch.log1p(pred_clamped)

    # theta layout from BobGRUVerbatim: [VTX, kdm, VTL, kmt, kmatm, R, lam, lamO]
    VTX_max = theta[..., 0:1]
    kdm = theta[..., 1:2]
    VTL_max = theta[..., 2:3]
    kmt = theta[..., 3:4]
    kmtm = theta[..., 4:5]
    R = theta[..., 5:6]

    mean_st = y_seq_3.mean(dim=-1, keepdim=True)
    VTX_raw = VTX_max / 0.125
    kdm_raw = kdm / 0.01
    VTL_raw = VTL_max / 0.075
    kmt_raw = kmt / 0.00035
    kmtm_raw = kmtm / 0.0035
    R_raw = R / 1.0

    VTX_state = (VTX_raw * R_raw * mean_st).clamp_min(0.0)
    kdm_state = (kdm_raw * mean_st).clamp_min(0.0)
    VTL_state = (VTL_raw * R_raw * mean_st).clamp_min(0.0)
    kmt_state = (kmt_raw * mean_st).clamp_min(0.0)
    kmtm_state = (kmtm_raw * mean_st).clamp_min(0.0)
    R_state = (R_raw * mean_st).clamp_min(0.0)

    log_VTX = torch.log1p(VTX_state)
    log_kdm = torch.log1p(kdm_state)
    log_VTL = torch.log1p(VTL_state)
    log_kmt = torch.log1p(kmt_state)
    log_kmtkm = torch.log1p(kmtm_state)
    log_R = torch.log1p(R_state)

    log_pred_all = torch.cat([log_pred, log_VTX, log_kdm, log_VTL, log_kmt, log_kmtkm, log_R], dim=-1)  # (B,K,9)
    log_zero = torch.zeros_like(log_VTX)
    log_y_all = torch.cat([log_y, log_zero, log_zero, log_zero, log_zero, log_zero, log_zero], dim=-1)  # (B,K,9)

    t_grid = torch.arange(K, device=device)
    valid_mask = (t_grid[None, :] < lengths[:, None]).unsqueeze(-1)  # (B,K,1)
    cut_idx = (lengths.to(torch.float32) * 0.95).to(torch.long)
    tail_mask = (t_grid[None, :] >= cut_idx[:, None]) & valid_mask.squeeze(-1)  # (B,K)

    # time weighting for p* (pm): second half gets weight N
    N = 3
    half_idx = lengths // 2
    is_second_half = (t_grid[None, :] >= half_idx[:, None]).unsqueeze(-1)  # (B,K,1)
    w_time = torch.where(
        is_second_half,
        torch.full_like(valid_mask, float(N), dtype=log_y_all.dtype),
        torch.ones_like(valid_mask, dtype=log_y_all.dtype),
    ) * valid_mask.to(torch.float32)

    w = torch.zeros_like(log_y_all, dtype=dtype)
    w[..., 0] = valid_mask.squeeze(-1)                     # mRNA
    w[..., 1] = tail_mask.to(dtype)                        # p (ignored by loss_weight but kept verbatim)
    w[..., 2] = (w_time * valid_mask).squeeze(-1)          # p*
    w[..., 3] = tail_mask.to(dtype)                        # VTX
    w[..., 4] = tail_mask.to(dtype)                        # kdm
    w[..., 5] = tail_mask.to(dtype)                        # VTL
    w[..., 6] = tail_mask.to(dtype)                        # kmt
    w[..., 7] = tail_mask.to(dtype)                        # kmtm
    w[..., 8] = tail_mask.to(dtype)                        # R proxy

    last_indices = (lengths - 1).clamp_min(0)
    batch_idx = torch.arange(B, device=device)
    y_last = log_y_all[batch_idx, last_indices, 2]
    pred_last = log_pred_all[batch_idx, last_indices, 2]
    extra_mse_chan = ((pred_last - y_last) ** 2).mean(0, keepdim=True)  # (1,)

    err2 = (log_pred_all - log_y_all).pow(2) * w
    mse_chan = err2.sum((0, 1)) / w.sum((0, 1)).clamp_min(1)
    mse_chan = torch.cat([mse_chan, extra_mse_chan])  # (10,)

    loss_weight = torch.tensor([1.25, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1], device=device, dtype=mse_chan.dtype)
    weighted = loss_weight * mse_chan
    mask = loss_weight.ne(0)
    den = mask.sum().clamp_min(1).to(weighted.dtype)
    return weighted[mask].sum() / den


def loss_fn_per_species(
    pred: torch.Tensor,
    y_seq: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    use_log_loss: bool = True,
) -> torch.Tensor:
    if use_log_loss:
        # clamp_min(0) before log1p — model predictions for unobserved channels
        # (e.g. R, O) can drift negative during training; log1p(<-1) → NaN and
        # corrupts the per-species diagnostic output. Doesn't affect the main
        # loss path, which is loss_fn with its own clamp_min.
        pred  = torch.log1p(pred.clamp_min(0.0))
        y_seq = torch.log1p(y_seq.clamp_min(0.0))
    se = (pred - y_seq).pow(2)
    if lengths is not None:
        mask = _build_loss_mask(lengths, se.shape[1], se.device)  # (B,K)
        se = se * mask.unsqueeze(-1)
        return se.sum(dim=(0, 1)) / mask.sum()
    return se.mean(dim=(0, 1))


# --- Normalization utilities ---
_NORM_STD_EPS = 1e-8


def _compute_zscore_stats(
    y0: np.ndarray, y_seq: np.ndarray,
    train_idx: np.ndarray, lengths: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std from training samples only (valid timesteps only)."""
    train_y0 = y0[train_idx]
    if lengths is not None:
        parts = [train_y0] + [y_seq[i, :lengths[i]] for i in train_idx]
    else:
        parts = [train_y0, y_seq[train_idx].reshape(-1, y0.shape[-1])]
    all_vals = np.concatenate(parts, axis=0)
    mean = all_vals.mean(axis=0).astype(np.float32)
    std  = np.maximum(all_vals.std(axis=0), _NORM_STD_EPS).astype(np.float32)
    return mean, std


def _apply_norm(
    arr: np.ndarray, method: str,
    mean: np.ndarray | None = None,
    std:  np.ndarray | None = None,
) -> np.ndarray:
    if method == "log":
        return np.log1p(np.clip(arr, 0.0, None)).astype(np.float32)
    if method == "sqrt":
        return np.sqrt(np.clip(arr, 0.0, None)).astype(np.float32)
    if method == "zscore":
        return ((arr - mean) / std).astype(np.float32)
    raise ValueError(f"Unknown obs_normalization: {method!r}. Choose from: log, sqrt, zscore")


def _endpoint_values(
    y_seq: np.ndarray,
    lengths: np.ndarray | None,
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


def _stratified_yield_split(
    pool_idx: np.ndarray,
    y_seq: np.ndarray,
    lengths: np.ndarray | None,
    n_val: int,
    n_test: int,
    stratify_bins: int,
    stratify_targets: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Yield-stratified train/val/test split on a subset of indices.

    Factored out so it can be called once over the whole dataset (legacy
    behaviour) or independently per source (when stratify_by_source=True).
    """
    if n_test + n_val >= len(pool_idx):
        raise ValueError(f"val={n_val}+test={n_test} >= pool size {len(pool_idx)}")
    sub_y = y_seq[pool_idx]
    sub_L = lengths[pool_idx] if lengths is not None else None
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


def _make_split_indices(
    *,
    N: int,
    y_seq: np.ndarray,
    lengths: np.ndarray | None,
    n_val: int,
    n_test: int,
    split_seed: int,
    stratified_split: bool,
    stratify_bins: int,
    stratify_targets: list[int] | None,
    z_expr: np.ndarray | None = None,
    stratify_z_expr: bool = False,
    test_real_only: bool = False,
    source_idx: np.ndarray | None = None,
    stratify_by_source: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create train/val/test indices; optionally stratified by endpoint bins
    and/or by the dataset's z_expr (real vs synth no-go) class flag.

    test_real_only: when True (and the dataset carries z_expr), val/test draws
    are restricted to real samples (z_expr==1) only. Train gets remaining real
    + all synth. Lets us train on (real + synth) but evaluate honestly on real.
    """
    if n_test + n_val >= N:
        raise ValueError(f"val_n={n_val} + test_n={n_test} >= N={N}")

    rng = np.random.default_rng(split_seed)
    all_idx = np.arange(N, dtype=np.int64)

    # test_real_only takes precedence: restrict val/test pool to real samples,
    # apply (optional) endpoint stratification within that pool, dump all synth
    # into train.
    if test_real_only and z_expr is not None:
        real_idx = np.where(z_expr == 1)[0]
        synth_idx = np.where(z_expr == 0)[0]
        n_real = len(real_idx)
        if n_test + n_val >= n_real:
            raise ValueError(
                f"test_real_only=True but val_n={n_val} + test_n={n_test} >= "
                f"n_real={n_real}. Reduce val_n/test_n or disable the flag."
            )

        # When stratify_by_source is also set, split OLD and NEW pools
        # independently within the real pool — same logic as the standalone
        # stratify_by_source path, but scoped to z_expr==1 samples only.
        # This produces the same test set as real_only runs so combined vs
        # real_only comparisons evaluate on identical samples.
        if (stratify_by_source and source_idx is not None
                and stratified_split and stratify_targets):
            P = int(y_seq.shape[-1])
            targets = [int(t) for t in stratify_targets if 0 <= int(t) < P]
            if targets:
                real_src = source_idx[real_idx]
                sources_present = sorted(int(s) for s in np.unique(real_src) if int(s) >= 0)
                if len(sources_present) >= 2:
                    pools = {s: real_idx[real_src == s] for s in sources_present}
                    sizes_arr = np.array([len(pools[s]) for s in sources_present], dtype=np.int64)
                    take_test_per_src = _allocate_counts(sizes_arr, int(n_test))
                    take_val_per_src  = _allocate_counts(sizes_arr, int(n_val))
                    tr_parts, va_parts, te_parts = [], [], []
                    src_name = {0: "old", 1: "new", 2: "synth"}
                    for src, n_te, n_va in zip(sources_present, take_test_per_src, take_val_per_src):
                        pool = pools[src]
                        tr_s, va_s, te_s = _stratified_yield_split(
                            pool, y_seq, lengths, int(n_va), int(n_te),
                            stratify_bins, targets, rng,
                        )
                        tr_parts.append(tr_s); va_parts.append(va_s); te_parts.append(te_s)
                        print(f"  test_real_only+stratify_by_source[{src_name.get(src, str(src))}]: "
                              f"pool={len(pool)} → train={len(tr_s)} val={len(va_s)} test={len(te_s)}")
                    real_train = np.concatenate(tr_parts)
                    val_idx   = np.concatenate(va_parts)
                    test_idx  = np.concatenate(te_parts)
                    train_idx = np.concatenate([real_train, synth_idx])
                    rng.shuffle(test_idx); rng.shuffle(val_idx); rng.shuffle(train_idx)
                    print(f"test_real_only: train={len(real_train)} real + {len(synth_idx)} synth; "
                          f"val={len(val_idx)} real; test={len(test_idx)} real")
                    return train_idx, val_idx, test_idx
                # Only one source in real pool → fall through to non-source path below.

        if stratified_split and stratify_targets:
            # Endpoint-stratify within the real pool only (no per-source split).
            P = int(y_seq.shape[-1])
            targets = [int(t) for t in stratify_targets if 0 <= int(t) < P]
            if not targets:
                raise ValueError("test_real_only + stratified_split: empty stratify_targets")
            real_lengths = lengths[real_idx] if lengths is not None else None
            vals = _endpoint_values(y_seq[real_idx], real_lengths, targets)
            std = vals.std(axis=0)
            keep = std > 1e-12
            if not np.any(keep):
                rng.shuffle(real_idx)
                test_idx = real_idx[:n_test]
                val_idx = real_idx[n_test:n_test + n_val]
                real_train = real_idx[n_test + n_val:]
            else:
                vals = vals[:, keep]
                ranks = np.empty_like(vals, dtype=np.float64)
                for j in range(vals.shape[1]):
                    order = np.argsort(vals[:, j], kind="mergesort")
                    inv = np.empty_like(order)
                    inv[order] = np.arange(vals.shape[0])
                    ranks[:, j] = inv / max(1, vals.shape[0] - 1)
                score = ranks.mean(axis=1)
                labels = _quantile_bin_1d(score, int(stratify_bins)).astype(np.int64)
                uniq = np.unique(labels)
                strata = []
                for u in uniq:
                    idx_u = real_idx[labels == u]
                    rng.shuffle(idx_u)
                    strata.append(idx_u)
                sizes = np.array([len(s) for s in strata], dtype=np.int64)
                take_test = _allocate_counts(sizes, int(n_test))
                test_parts, rem_parts = [], []
                for s, k in zip(strata, take_test):
                    test_parts.append(s[:k]); rem_parts.append(s[k:])
                test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=np.int64)
                rem_strata_sizes = np.array([len(r) for r in rem_parts], dtype=np.int64)
                take_val = _allocate_counts(rem_strata_sizes, int(n_val))
                val_parts, train_real_parts = [], []
                for r, k in zip(rem_parts, take_val):
                    val_parts.append(r[:k]); train_real_parts.append(r[k:])
                val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
                real_train = np.concatenate(train_real_parts) if train_real_parts else np.array([], dtype=np.int64)
        else:
            rng.shuffle(real_idx)
            test_idx = real_idx[:n_test]
            val_idx = real_idx[n_test:n_test + n_val]
            real_train = real_idx[n_test + n_val:]

        train_idx = np.concatenate([real_train, synth_idx])
        rng.shuffle(test_idx); rng.shuffle(val_idx); rng.shuffle(train_idx)
        print(f"test_real_only: train={len(real_train)} real + {len(synth_idx)} synth; "
              f"val={len(val_idx)} real; test={len(test_idx)} real")
        return train_idx, val_idx, test_idx

    # z_expr stratification: split real and synth pools separately and allocate
    # n_val / n_test proportionally to each. This guarantees val/test contain
    # both classes in roughly the dataset's ratio (so metrics like R² aren't
    # computed on ~99% synthetic flat trajectories).
    if stratify_z_expr:
        if z_expr is None:
            raise ValueError("stratify_z_expr=True but dataset has no z_expr label.")
        real_idx = np.where(z_expr == 1)[0]
        synth_idx = np.where(z_expr == 0)[0]
        rng.shuffle(real_idx)
        rng.shuffle(synth_idx)

        # Allocate test count proportionally between real and synth pools.
        n_test_real = int(round(n_test * len(real_idx) / max(N, 1)))
        n_test_synth = max(0, n_test - n_test_real)
        n_val_real = int(round(n_val * len(real_idx) / max(N, 1)))
        n_val_synth = max(0, n_val - n_val_real)

        # Don't over-draw from either pool.
        n_test_real = min(n_test_real, len(real_idx))
        n_test_synth = min(n_test_synth, len(synth_idx))
        n_val_real = min(n_val_real, max(0, len(real_idx) - n_test_real))
        n_val_synth = min(n_val_synth, max(0, len(synth_idx) - n_test_synth))

        test_idx = np.concatenate([real_idx[:n_test_real], synth_idx[:n_test_synth]])
        val_idx = np.concatenate([
            real_idx[n_test_real:n_test_real + n_val_real],
            synth_idx[n_test_synth:n_test_synth + n_val_synth],
        ])
        train_idx = np.concatenate([
            real_idx[n_test_real + n_val_real:],
            synth_idx[n_test_synth + n_val_synth:],
        ])
        rng.shuffle(test_idx); rng.shuffle(val_idx); rng.shuffle(train_idx)
        print(f"stratify_z_expr: test={n_test_real} real + {n_test_synth} synth; "
              f"val={n_val_real} real + {n_val_synth} synth; "
              f"train={(z_expr[train_idx]==1).sum()} real + {(z_expr[train_idx]==0).sum()} synth")
        return train_idx, val_idx, test_idx

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

    # Source-aware stratification: split OLD and NEW pools independently via the
    # yield-bin logic so each (source × yield_bin) cell is matched across train/
    # val/test. Counts allocated proportional to each source's pool size so the
    # source-mix in val/test matches the dataset's natural ratio.
    if stratify_by_source and source_idx is not None:
        sources_present = sorted(int(s) for s in np.unique(source_idx) if int(s) >= 0)
        if len(sources_present) >= 2:
            pools = {s: np.where(source_idx == s)[0] for s in sources_present}
            sizes_arr = np.array([len(pools[s]) for s in sources_present], dtype=np.int64)
            take_test_per_src = _allocate_counts(sizes_arr, int(n_test))
            take_val_per_src  = _allocate_counts(sizes_arr, int(n_val))
            tr_parts, va_parts, te_parts = [], [], []
            src_name = {0: "old", 1: "new", 2: "synth"}
            for src, n_te, n_va in zip(sources_present, take_test_per_src, take_val_per_src):
                pool = pools[src]
                tr_s, va_s, te_s = _stratified_yield_split(
                    pool, y_seq, lengths, int(n_va), int(n_te),
                    stratify_bins, targets, rng,
                )
                tr_parts.append(tr_s); va_parts.append(va_s); te_parts.append(te_s)
                print(f"  stratify_by_source[{src_name.get(src, str(src))}]: "
                      f"pool={len(pool)} → train={len(tr_s)} val={len(va_s)} test={len(te_s)}")
            train_idx = np.concatenate(tr_parts)
            val_idx   = np.concatenate(va_parts)
            test_idx  = np.concatenate(te_parts)
            rng.shuffle(test_idx); rng.shuffle(val_idx); rng.shuffle(train_idx)
            return train_idx, val_idx, test_idx

    vals = _endpoint_values(y_seq, lengths, targets)
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


def _resolve_split(cfg, ds, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single entry point for train/val/test split selection.

    Precedence (first match wins):
      1. `fixed_split_file`     — manual triple loaded from an `.npz` file.
      2. `fixed_test_idx_path`  — test set from `.npy`; train/val random from the rest.
      3. default               — count-based random split (`val_n`, `test_n`),
                                  optionally stratified by endpoint quantiles.
    """
    if cfg.fixed_split_file:
        train_idx, val_idx, test_idx = _load_fixed_split_triples(
            cfg.fixed_split_file, int(cfg.fixed_split_idx), ds.original_indices,
        )
        for name, arr in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
            if arr.size and (arr.max() >= N or arr.min() < 0):
                raise ValueError(
                    f"fixed_split {name}_idx out of range for N={N} "
                    f"(min={arr.min()}, max={arr.max()})"
                )
        print(
            f"Split: fixed_split_file={cfg.fixed_split_file}"
            f" | idx={int(cfg.fixed_split_idx)}"
            f" | n_test={len(test_idx)} n_val={len(val_idx)} n_train={len(train_idx)}"
        )
        return train_idx, val_idx, test_idx

    n_test = int(cfg.test_n) if cfg.test_n > 0 else 0
    n_val  = int(cfg.val_n)  if cfg.val_n  > 0 else max(1, int(N * cfg.val_frac))

    if cfg.fixed_test_idx_path:
        fixed_test = np.load(cfg.fixed_test_idx_path).astype(np.int64)
        if fixed_test.max() >= N or fixed_test.min() < 0:
            raise ValueError(f"fixed_test_idx out of range for dataset size {N}")
        remaining = np.setdiff1d(np.arange(N, dtype=np.int64), fixed_test)
        rng = np.random.default_rng(int(cfg.split_seed))
        rng.shuffle(remaining)
        if n_val > len(remaining):
            raise ValueError(f"val_n={n_val} exceeds remaining {len(remaining)}")
        val_idx = remaining[:n_val]
        train_idx = remaining[n_val:]
        test_idx = fixed_test
        print(
            f"Split: fixed_test_idx_path={cfg.fixed_test_idx_path}"
            f" | n_test={len(test_idx)} n_val={len(val_idx)} n_train={len(train_idx)}"
        )
        return train_idx, val_idx, test_idx

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
        z_expr=getattr(ds, "z_expr", None),
        stratify_z_expr=bool(cfg.stratify_z_expr),
        test_real_only=bool(cfg.test_real_only),
        source_idx=getattr(ds, "source_idx", None),
        stratify_by_source=bool(getattr(cfg, "stratify_by_source", False)),
    )
    kind = ("stratified+z_expr" if cfg.stratify_z_expr and cfg.stratified_split
            else "stratified" if cfg.stratified_split
            else "z_expr" if cfg.stratify_z_expr
            else "random")
    extra = (f" | bins={int(cfg.stratify_bins)}"
             f" | targets={cfg.stratify_targets if cfg.stratify_targets is not None else 'auto'}"
             if cfg.stratified_split else "")
    print(
        f"Split: {kind} | seed={int(cfg.split_seed)}{extra}"
        f" | n_test={len(test_idx)} n_val={len(val_idx)} n_train={len(train_idx)}"
    )
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
    # If set, load test indices from this .npy file (overrides random/stratified
    # test selection). Train/val are then drawn from the remaining samples.
    fixed_test_idx_path: str | None = None
    train_n: int = 0   # cap on train samples (0 = use all remaining after val/test)
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
    # Stratify train/val/test by the dataset's `z_expr` flag (real vs synth no-go).
    # When True, val_n and test_n are allocated proportionally between real and
    # synthetic samples so each split preserves the dataset's class ratio. This
    # is what you want for txtl_combined.npz so val/test aren't ~99% synthetics.
    stratify_z_expr: bool = False
    # Honest eval on real-only: when True and z_expr is present, val/test are
    # drawn only from real samples (z_expr==1); train gets the remaining real
    # + all synth. Use for datasets that mix synth bulk into training but where
    # the deployment distribution is real-only. Overrides stratify_z_expr's
    # proportional mixing for the val/test side.
    test_real_only: bool = False
    # Ablation knob: drop synthetic no-go rows from the dataset before training.
    # Default True (keep them — matches existing behavior). Set False to A/B test
    # whether the synth bulk + lambda_zero_traj loss actually helps real-data fit.
    use_synthetic_data: bool = True

    num_workers: int = 0
    pin_memory: bool = True

    scaffold: str = "reduced5"
    hidden: int = 128
    lift_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    theta_lo: float = 1e-3
    theta_hi: float = 2.0
    n_substeps: int = 1

    ff_mult: int = 2  # feedforward multiplier inside each transformer layer (2=current, 4=standard)

    # Transformer-specific (ignored by non-Transformer models via **kwargs)
    context_len: int = 64  # sliding window size; set >= sequence length for full context
    tf_group_size: int = 32  # grouped-TF chunk size for ode_transformer_grouped
    ar_gap: int = 4  # autoregressive steps inserted between grouped-TF chunks
    # Transformer ablation knobs
    use_absolute_pos: bool = False     # window-relative pos-embed (False) vs absolute trajectory pos (True)
    max_seq_len: int = 512             # embedding table size when use_absolute_pos=True
    grad_checkpoint: bool = False      # checkpoint the per-step transformer call to cut activation memory
    append_time_feature: bool = False  # append normalized cumulative time as an extra encoder input

    # Mamba-specific (ignored by non-Mamba models via **kwargs)
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4

    # neural_ode_correction / fixed_theta_nn-specific (ignored by other models via **kwargs)
    nn_hidden: int = 256
    nn_layers: int = 2

    forget_bias_init: Optional[float] = None  # None = PyTorch default; 1.0 = Gers/Jozefowicz positive shift
    slstm_drop_last_layer: bool = True         # ode_slstm: True = dropout every layer (match GRU); False = original (between layers)
    legacy_forget_bias_bug: bool = False      # reproduce pre-fix fill_(0.0) on both bias_ih and bias_hh

    # Plain MSE on (mm, pm) in raw scale, no normalization, no aux terms.
    # Requires the model to emit an 8-d theta layout
    # [VTX, kdm, VTL, kmt, kmatm, R, lam, lamO] — produced by any model paired
    # with scaffold=ivtt_analytic.
    use_ivtt_mse_loss: bool = False

    # Manual split: load one of the (train, val, test) triples stored in an
    # `.npz` file (keys: train_i / val_i / test_i). When set, this overrides
    # all other split options including `fixed_test_idx_path`.
    fixed_split_file: Optional[str] = None
    fixed_split_idx: int = 0

    use_basal: bool = False
    beta_regularization: bool = False
    lambda_beta: float = 1.0

    theta_bounded: bool = True   # if False, use softplus (unbounded above) instead of gamma

    # OdeRNN encoder knobs — defaults preserve current behaviour for synthetic data.
    lift_skip: bool = False               # drop the lift MLP and feed feat→GRU directly (Bob)
    gru_variant: str = "nn_gru"           # "nn_gru" | "stacked_cell" (Bob's stacked GRUCell + dropout-on-last)
    gru_init: str = "default"             # "default" | "supervisor" (orthogonal_ + xavier_ + zeros)
    head_init: str = "default"            # "default" | "supervisor" (xavier_ + zeros, unconditional)
    y0_theta_init: bool = False           # ode_rnn: add MLP(y0) bias to theta-head logits at every step
    encoder_use_time: bool = False        # ode_rnn: concat τ_k = k/(K-1) ∈ [0,1] to encoder feat (Experiment A
                                          # in new_scaffolds.tex §3.1 — "normalized time as encoder input").
    encoder_use_log_dt: bool = False      # ode_rnn: concat log(dt_k) to encoder feat (dt-awareness for variable
                                          # grids — OLD 60s vs NEW ~600s).
    theta_head_transform: str = "log_gamma"  # "log_gamma" | "gamma"
    theta_head_tau: float = 1.0           # log_gamma sigmoid temperature (Bob: 2.3)
    u_transform: str = "none"             # forward-time u feature transform ("none" | "sqrt" | "cumsum" | …)
    y_transform: str = "none"             # forward-time y feature transform ("none" | "sqrt_clamp1" | "log1p" | …)

    grad_clip: float = 1.0
    teacher_forcing: bool = True
    tf_every: int = 50
    tf_drop_epoch: int = 10**9

    # checkpointing cadence (0 disables periodic ckpts)
    ckpt_every: int = 10

    # If True, run endpoint R² analysis at end of training and save the plot
    # + cache into exp_dir. Also picked up by replot.py post-hoc.
    endpoint_r2: bool = False

    l1_regularization: bool = False   # smoothness: penalizes mean |theta[t] - theta[t-1]|
    l2_regularization: bool = False   # smoothness: penalizes mean (theta[t] - theta[t-1])^2

    lambda_reg: float = 0.001

    # If set (e.g. [0, 12]), supervise loss/TF only on those species indices.
    # If null/None, supervises all observed species (default behaviour).
    obs_idx: list[int] | None = None

    # If set, restrict GRU encoder input to these species indices (must match obs_idx
    # for a strictly partial-observation experiment). None = all P species (default).
    gru_y_cols: list[int] | None = None

    # If set, restrict GRU encoder input to these control (u) column indices.
    # Used e.g. by the IVTT analytic scaffold to drop the DNA c column from the
    # encoder feature (DNA c is consumed by the scaffold via dna_cum_total instead).
    # None = all U columns (default).
    gru_u_cols: list[int] | None = None

    # If set, slice ds.y0 and ds.y_seq to these column indices after loading.
    # Lets you use a dataset with more species than the scaffold expects by
    # selecting the subset (in scaffold state order) before the P_obs check.
    # Example: [0, 11, 12, 3, 4, 1, 2] picks the 7 kovacs_7 species from the
    # 14-species pulsed dataset.
    dataset_species_subset: list[int] | None = None

    # When True, compute loss only at the first and last time step (start/end
    # supervision). Useful to test whether models generalise from endpoint-only
    # signal. Not supported for variable-length sequences.
    supervise_endpoints_only: bool = False

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

    # 'ode_rnn' (default), 'ode_rnn_2020' (latent ODE-RNN style),
    # or 'neural_ode' (pure MLP baseline)
    model_class: str = "ode_rnn"

    # Pre-normalise y0/y_seq before training. Options: "log", "sqrt", "zscore", null.
    # When set, the internal log1p loss is replaced with plain MSE.
    obs_normalization: str | None = None

    # --- new-feature options (defaults preserve all previous behaviour) ---

    # Bob's "gate the observation": per-sample, per-channel min-subtraction on
    # y0/y_seq so failure/baseline traces sit near 0. Applied once at dataset
    # load (Dataset.__init__), no per-batch overhead. False = unchanged.
    subtract_channel_min: bool = False
    # Channels to gate when subtract_channel_min=True. None = all P channels.
    # Typical for the IVTT 7-state layout: [3, 5] (mm = Broccoli, pm = mCherry/2).
    subtract_channel_min_cols: list[int] | None = None

    # Stratify the train/val/test split jointly on (source × yield_bin) so
    # each (OLD, NEW) population gets the same yield-bin distribution in
    # train / val / test. Without this, the existing yield-only stratification
    # can let NEW high-yield outliers cluster in test by chance (with only
    # ~30 NEW-test samples, this happens easily).
    stratify_by_source: bool = False

    # Per-source loss weighting. Dict {source_name: multiplier} where source_name
    # is "old"/"new"/"synth". Per-sample weight is applied as a scalar multiplier
    # on each sample's contribution to the masked loss. Default None = uniform.
    # Example: {"old": 1.0, "new": 5.0} upweights NEW 5× so the model can't
    # satisfy the loss by fitting OLD only.
    source_loss_weights: dict | None = None

    # Class-balanced sampling. When True, builds a WeightedRandomSampler over
    # the train split with weights inversely proportional to per-source count
    # so each minibatch sees a roughly 50/50 mix of OLD/NEW (and synth, when
    # present). Use INSTEAD OF source_loss_weights, not on top, unless you
    # specifically want both (rare).
    balance_source_sampler: bool = False

    # Hard cap on synth rows at dataset load time. Randomly subsamples synth to
    # at most this many rows (all real rows kept). More efficient than loss
    # downweighting — no wasted forward/backward on discarded synth samples.
    # E.g. synth_max_samples: 150 ≈ synth_w0p05 expected gradient but ~20× faster
    # per epoch. None = keep all synth (default). Seeded by cfg.seed for repro.
    synth_max_samples: int | None = None

    # Per-channel suppression for synth (source==2) samples.
    # List of RAW obs-channel indices (same coordinate as obs_idx) to zero in
    # the trajectory loss and endpoint loss for any sample with source_idx==2.
    # Synth no-expression trajectories have pm≈0 for every timestep, which
    # biases the model toward predicting low protein on real expressing samples.
    # Setting synth_obs_mask_channels: [5]  (where 5 is the raw pm index) lets
    # synth samples contribute mRNA / structural gradients while never teaching
    # the model that "protein should be zero."
    # None (default) = no per-channel masking, all channels supervised uniformly.
    synth_obs_mask_channels: list | None = None

    # Diagnostics: if >0, plot_predictions also dumps this many extra sample
    # plots restricted to NEW-source samples (source_label=='new') from the
    # test split, with a "new_" filename prefix. Useful when training on
    # mixed OLD+NEW native-grid NPZs and you want to eyeball post-tube-opening
    # behaviour without manually picking indices.
    plot_extra_new_samples: int = 0

    # "Gate the parameter" (Bob's ablation knob): pin chosen entries of θ(t) to
    # constant values by collapsing their (theta_lo, theta_hi) box to a single
    # point. Dict of {theta_idx: value}. Empty/None = no pinning (default).
    # Example:  pin_theta: {0: 0.0}  pins θ[0] to 0.
    pin_theta: dict[int, float] | None = None

    # Override per-parameter bounds without touching the scaffold.
    # Dict of {theta_idx: new_hi}. Useful for testing different K-constant ranges
    # without editing scaffolds.py.  Example (M8 K-bounds sweep):
    #   theta_hi_override: {6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1, 10: 0.1, 11: 0.1}
    theta_hi_override: dict[int, float] | None = None

    # K-anchor sparse-θ readout (tex Models B2/B3). When using model_class
    # "ode_rnn_sparse_theta", set n_theta_anchors (typical 1, 3, or 6). None =
    # dense per-step θ (default; pair with the regular ode_rnn / ode_rnn_basal_v2).
    n_theta_anchors: int | None = None
    # Interpolation between anchors: "piecewise" (B2) or "linear" (B3).
    anchor_interp: str = "piecewise"
    # V2-only: how the K anchors are placed.
    #   "uniform" — linspace(0, T-1, n_theta_anchors) (the tex default).
    #   "bolus"   — per-sample anchors at every bolus event in u_seq (plus k=0);
    #               n_theta_anchors is ignored for placement but still required
    #               (>=1) as a sanity-check value. Use bolus_max_anchors to cap.
    anchor_mode: str = "uniform"
    # Cap on bolus-mode anchors per sample (0 = no cap). Late boluses past the
    # cap still feed the GRU encoder but don't trigger a new θ anchor.
    bolus_max_anchors: int = 0

    # Model 7 zero-trajectory boundary loss weight. Penalises predicted observed
    # values on samples with z_expr == 0 (synthetic no-go) toward zero. 0 =
    # disabled (default). Typical range 1e-3 – 1e-1. No-op on legacy datasets.
    lambda_zero_traj: float = 0.0

    # --- supervisor-style composable loss knobs (default None → unchanged) ---
    # Per-species multipliers keyed by raw state index. Example for mm=3, pm=5:
    #   species_weights: {3: 1.25, 5: 1.0}
    species_weights: dict | None = None
    # Per-species time upweighting keyed by raw state index. Value is
    # [start_frac, factor]: for t >= int(start_frac · length), multiply that
    # channel's weight by factor. Supervisor uses pm second-half ×3:
    #   time_upweight: {5: [0.5, 3.0]}
    time_upweight: dict | None = None
    # Extra MSE term evaluated only at the final valid timestep on the given
    # raw state indices. Supervisor uses pm endpoint with weight 0.1:
    #   lambda_endpoint: 0.1
    #   endpoint_channels: [5]
    lambda_endpoint: float = 0.0
    endpoint_channels: list[int] | None = None
    # Comparison space for the endpoint term: "log" (default, original behaviour),
    # "sqrt" (penalise high-yield misses ~proportionally — counters log1p tail
    # under-prediction), or "linear" (full high-end weight; can be unstable).
    endpoint_loss_space: str = "log"
    # PEAK term: MSE on the per-trajectory MAX of `peak_channels` (default off).
    # Anchors mRNA-max (the R²(mRNA-max) metric) so the model is supervised on
    # both cascade ends (mRNA-max + protein-final). Shares loss_clamp_min and
    # endpoint_loss_space. lambda_peak=0 disables (default → no behaviour change).
    lambda_peak: float = 0.0
    peak_channels: list[int] | None = None
    # Supervisor's loss divides the final composite by the number of nonzero
    # loss-weight channels (=3 in his recipe: mm + pm-body + pm-endpoint).
    # Set this to that count to reproduce his /N normaliser exactly; leave None
    # to skip (relative gradient direction is unchanged, only the overall scale).
    # Tracked as a knob so we can ablate the normaliser choice later.
    loss_normalizer_channels: int | None = None
    # Floor pred/y at this value before log1p inside loss_fn / endpoint_mse.
    # Supervisor parity: 1.0 (matches loss_fn_ivtt_mse's clamp_min(1.0)). 0.0
    # disables (plain log1p). Creates a "dead zone" for small values where
    # mismatches don't contribute to the loss — useful when many trajectory
    # segments are structurally near zero (pm early in IVTT traces).
    loss_clamp_min: float = 0.0
    # ── L_oxygen (Model 9 / tube-opening experiments) ──────────────────────
    # Tex "Common loss functions / Oxygen loss":
    #   L_oxygen = ρ Σ_{i ∈ D_open} Σ_{t_n > t_open,i} (P̂_fluor,i − P_obs,i)²
    # Adds an extra squared error on the fluorescent-protein channel
    # restricted to timesteps AFTER each sample's tube-opening event.
    # No-op when lambda_oxygen=0 or when the dataset has no `u_open` channel.
    # oxygen_protein_channel: raw dataset state index for P_fluor (default 5 = pm).
    # oxygen_u_open_name:     control_names entry identifying the opening signal.
    lambda_oxygen: float = 0.0
    oxygen_protein_channel: int = 5
    oxygen_u_open_name: str = "u_open"


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
        ("plots/theta_sample0", plots_dir / "theta_sample0.png"),
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

    ds = ODEDataset(cfg.dataset_path, use_synthetic_data=bool(cfg.use_synthetic_data),
                    synth_max_samples=cfg.synth_max_samples,
                    seed=int(cfg.seed))

    if cfg.dataset_species_subset is not None:
        idx = np.array(cfg.dataset_species_subset, dtype=np.int64)
        ds.obs_names   = ds.obs_names[idx]   if ds.obs_names   is not None else None
        ds.obs_indices = ds.obs_indices[idx]
        ds.y0          = ds.y0[:, idx]
        ds.y_seq       = ds.y_seq[:, :, idx]
        names_str = list(ds.obs_names) if ds.obs_names is not None else idx.tolist()
        print(f"Species subset: {names_str} ({len(idx)} of original species)")

    N = len(ds)

    train_idx, val_idx, test_idx = _resolve_split(cfg, ds, N)

    if int(cfg.train_n) > 0 and int(cfg.train_n) < len(train_idx):
        train_idx = train_idx[: int(cfg.train_n)]
        print(f"Data-scarce: capped train to train_n={int(cfg.train_n)}")

    # persist split so plotting always uses the correct test indices
    np.savez(exp_dir / "split.npz",
             train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    # --- optional pre-normalisation (stats from train split only) ---
    use_log_loss = cfg.obs_normalization is None
    if cfg.obs_normalization:
        method = cfg.obs_normalization
        if method not in ("log", "sqrt", "zscore"):
            raise ValueError(f"obs_normalization must be one of: log, sqrt, zscore (got {method!r})")
        norm_mean = norm_std = None
        if method == "zscore":
            norm_mean, norm_std = _compute_zscore_stats(
                ds.y0, ds.y_seq, train_idx,
                ds.lengths if ds.variable_length else None,
            )
        ds.y0   = _apply_norm(ds.y0,   method, norm_mean, norm_std)
        ds.y_seq = _apply_norm(ds.y_seq, method, norm_mean, norm_std)
        save_kwargs: dict = {}
        if norm_mean is not None:
            save_kwargs["mean"] = norm_mean
            save_kwargs["std"]  = norm_std
        np.savez(exp_dir / "norm_stats.npz", method=np.array(cfg.obs_normalization), **save_kwargs)
        print(f"Normalization: {method}" + (
            f" | per-channel stats from {len(train_idx)} training samples" if method == "zscore" else ""
        ))

    collate_fn = collate_varlen if ds.variable_length else collate

    # Optional class-balanced sampler: inverse-frequency weights per source so
    # each minibatch sees a balanced mix of OLD/NEW/synth. Falls back to plain
    # shuffle when the dataset has no source labels.
    _sampler = None
    _shuffle = True
    if bool(cfg.balance_source_sampler):
        if ds.source_idx is None:
            print("balance_source_sampler=True but dataset has no source_label; ignoring.")
        else:
            tr_src = ds.source_idx[train_idx.astype(np.int64)]
            counts = {int(s): int((tr_src == s).sum()) for s in np.unique(tr_src) if s >= 0}
            print(f"balance_source_sampler: train counts by source = {counts}")
            inv = {s: (1.0 / c if c > 0 else 0.0) for s, c in counts.items()}
            w = np.array([inv.get(int(s), 0.0) for s in tr_src], dtype=np.float64)
            _sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.from_numpy(w),
                num_samples=len(train_idx),
                replacement=True,
            )
            _shuffle = False

    train_loader = DataLoader(
        torch.utils.data.Subset(ds, train_idx.tolist()),
        batch_size=cfg.batch_size,
        shuffle=_shuffle,
        sampler=_sampler,
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

    # infer dims (item now returns (y0, u, y, dt, z) — 5 elements)
    item_ex = ds[0]
    y0_ex, u_ex = item_ex[0], item_ex[1]
    P_obs = int(y0_ex.shape[0])
    U = int(u_ex.shape[-1])

    scaffold = None
    if cfg.model_class != "bob_gru_verbatim":
        if cfg.scaffold not in SCAFFOLDS:
            raise ValueError(f"Unknown scaffold '{cfg.scaffold}'. Available: {list(SCAFFOLDS.keys())}")
        scaffold = SCAFFOLDS[cfg.scaffold]

        # "Gate the parameter" (cfg.pin_theta): collapse (lo, hi) box of selected
        # θ entries to a single point so the bounded readout returns the constant.
        # Copy the vectors so we don't mutate the shared SCAFFOLDS dict instance.
        if cfg.pin_theta:
            lo_vec = list(scaffold.theta_lo_vec) if scaffold.theta_lo_vec is not None \
                else [float(cfg.theta_lo)] * scaffold.theta_dim
            hi_vec = list(scaffold.theta_hi_vec) if scaffold.theta_hi_vec is not None \
                else [float(cfg.theta_hi)] * scaffold.theta_dim
            # log_gamma(x, lo, hi) = lo·exp(log(hi/lo)·σ(x)) can't represent
            # exactly 0 (log(0/0) = NaN). Clamp pinned values to a tiny epsilon
            # when log_gamma is the active transform — for ODE kinetics, 1e-12
            # is effectively zero over any realistic time horizon.
            _pin_eps = 1e-12 if str(cfg.theta_head_transform) == "log_gamma" else 0.0
            _pin_applied: dict[int, float] = {}
            for k, v in cfg.pin_theta.items():
                idx = int(k)
                if not (0 <= idx < scaffold.theta_dim):
                    raise ValueError(
                        f"pin_theta index {idx} out of range for scaffold "
                        f"'{cfg.scaffold}' (theta_dim={scaffold.theta_dim})"
                    )
                v_clamped = max(float(v), _pin_eps) if _pin_eps > 0 else float(v)
                lo_vec[idx] = v_clamped
                hi_vec[idx] = v_clamped
                _pin_applied[idx] = v_clamped
            scaffold.theta_lo_vec = lo_vec
            scaffold.theta_hi_vec = hi_vec
            if _pin_eps > 0 and any(float(v) < _pin_eps for v in cfg.pin_theta.values()):
                print(f"pin_theta: log_gamma can't represent exact 0; pinned values "
                      f"clamped at eps={_pin_eps:.0e}. Applied: {_pin_applied}")
            else:
                print(f"pin_theta: collapsed θ entries {sorted(cfg.pin_theta)} to constants {[cfg.pin_theta[k] for k in sorted(cfg.pin_theta)]}")

        if cfg.theta_hi_override:
            if not cfg.pin_theta:  # vectors not yet copied; copy now
                lo_vec = list(scaffold.theta_lo_vec) if scaffold.theta_lo_vec is not None \
                    else [float(cfg.theta_lo)] * scaffold.theta_dim
                hi_vec = list(scaffold.theta_hi_vec) if scaffold.theta_hi_vec is not None \
                    else [float(cfg.theta_hi)] * scaffold.theta_dim
            for k, v in cfg.theta_hi_override.items():
                idx = int(k)
                if not (0 <= idx < scaffold.theta_dim):
                    raise ValueError(
                        f"theta_hi_override index {idx} out of range for scaffold "
                        f"'{cfg.scaffold}' (theta_dim={scaffold.theta_dim})"
                    )
                hi_vec[idx] = float(v)
            scaffold.theta_lo_vec = lo_vec
            scaffold.theta_hi_vec = hi_vec
            print(f"theta_hi_override: updated θ hi bounds at indices {sorted(cfg.theta_hi_override)}")

        if scaffold.P != P_obs and (
            getattr(scaffold, "obs_state_idx", None) is None
            or getattr(scaffold, "control_state_map", None) is None
        ):
            raise ValueError(
                f"Scaffold {cfg.scaffold} expects P={scaffold.P}, but dataset has P_obs={P_obs}, "
                "and the scaffold did not declare obs_state_idx/control_state_map for the lift."
            )

    # Build u_to_y_jump:
    #   - If scaffold matches dataset (P == P_obs): use dataset's native indices.
    #   - If scaffold is partially observed: build a (U, scaffold.P) jump matrix
    #     by looking up each dataset control_name in scaffold.control_state_map.
    if scaffold.P == P_obs:
        u_to_y_jump = make_u_to_y_jump(ds.control_indices, ds.obs_indices, device=device)  # (U,P_obs)
    else:
        U_total = int(ds.control_indices.shape[0])
        u_to_y_jump = torch.zeros((U_total, scaffold.P), dtype=torch.float32, device=device)
        if ds.control_names is None:
            raise ValueError("Partial-observability lift requires dataset control_names; rebuild the npz.")
        for j, name in enumerate(list(ds.control_names)):
            target = scaffold.control_state_map.get(str(name).strip())
            if target is not None:
                # Support single int OR list[int] targets (e.g. Lysate → [AA, PEG])
                targets = list(target) if isinstance(target, (list, tuple)) else [target]
                for t in targets:
                    u_to_y_jump[j, int(t)] = 1.0
        print(f"Partial-observability lift: scaffold.P={scaffold.P}, P_obs={P_obs}, "
              f"obs_state_idx={scaffold.obs_state_idx}, "
              f"control_state_map={scaffold.control_state_map}")

    if cfg.model_class not in MODELS:
        raise ValueError(f"Unknown model_class '{cfg.model_class}'. Available: {list(MODELS.keys())}")
    # Any of the K-anchor sparse-θ wrappers accepts two extra kwargs. Only pass
    # them to those models so the other model classes' signatures stay untouched.
    SPARSE_THETA_MODELS = {
        "ode_rnn_sparse_theta",
        "ode_slstm_sparse_theta",
        "ode_rnn_basal_v2_sparse_theta",
    }
    # V2 is the single-pass sparse-θ model (piecewise interp only — its θ-head
    # fires K times instead of K-times-then-subsampled, so it's JIT-scriptable
    # and ~2× faster than the V1 two-pass wrapper).
    SPARSE_THETA_V2_MODELS = {"ode_rnn_sparse_theta_v2"}
    sparse_theta_kwargs = {}
    if cfg.model_class in SPARSE_THETA_MODELS:
        if cfg.n_theta_anchors is None:
            raise ValueError(
                f"model_class='{cfg.model_class}' requires cfg.n_theta_anchors (e.g. 1, 3, or 6)."
            )
        sparse_theta_kwargs = dict(
            n_theta_anchors=int(cfg.n_theta_anchors),
            anchor_interp=str(cfg.anchor_interp),
        )
    elif cfg.model_class in SPARSE_THETA_V2_MODELS:
        if cfg.n_theta_anchors is None:
            raise ValueError(
                f"model_class='{cfg.model_class}' requires cfg.n_theta_anchors (e.g. 1, 3, or 6)."
            )
        # anchor_interp is fixed at piecewise for V2 (linear needs lookahead).
        sparse_theta_kwargs = dict(
            n_theta_anchors=int(cfg.n_theta_anchors),
            anchor_mode=str(cfg.anchor_mode),
            bolus_max_anchors=int(cfg.bolus_max_anchors),
        )
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
        use_absolute_pos=cfg.use_absolute_pos,
        max_seq_len=cfg.max_seq_len,
        grad_checkpoint=cfg.grad_checkpoint,
        append_time_feature=cfg.append_time_feature,
        theta_bounded=cfg.theta_bounded,
        d_state=cfg.d_state,
        expand=cfg.expand,
        d_conv=cfg.d_conv,
        forget_bias_init=cfg.forget_bias_init,
        slstm_drop_last_layer=cfg.slstm_drop_last_layer,
        legacy_forget_bias_bug=cfg.legacy_forget_bias_bug,
        # Partial-observability: override the dataset-space gru_y_cols with the
        # scaffold's obs positions, so the encoder reads mm/pm at scaffold cols
        # regardless of how cfg.gru_y_cols was written.
        gru_y_cols=(scaffold.obs_state_idx
                    if (scaffold.P != P_obs and getattr(scaffold, "obs_state_idx", None) is not None)
                    else cfg.gru_y_cols),
        gru_u_cols=cfg.gru_u_cols,
        lift_skip=cfg.lift_skip,
        gru_variant=cfg.gru_variant,
        gru_init=cfg.gru_init,
        head_init=cfg.head_init,
        y0_theta_init=cfg.y0_theta_init,
        encoder_use_time=cfg.encoder_use_time,
        encoder_use_log_dt=cfg.encoder_use_log_dt,
        theta_head_transform=cfg.theta_head_transform,
        theta_head_tau=cfg.theta_head_tau,
        # Channel-expanding u_transforms (pulse_cumsum_sqrt / decay_trace /
        # cumsum_timesince_sqrt) must be known at construction to size the lift
        # layer; ode_rnn reads self.u_transform in forward. Other models ignore
        # it (absorbed by **kwargs) and use the forward-time u_transform arg.
        u_transform=cfg.u_transform,
        **sparse_theta_kwargs,
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
        except Exception as e:
            # Bare `except:` previously swallowed the error — the actual TorchScript
            # diagnostic is what tells you which line/branch is non-scriptable.
            import traceback
            print(f'JIT scripting failed: {type(e).__name__}: {e}')
            traceback.print_exc()
            print('Continuing with eager model.')

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

    # mech_names is used to label the per-species loss in the epoch log line.
    # Must match the SUPERVISED channels (len == len(cfg.obs_idx)), not the
    # full dataset obs_names list — otherwise zip() pairs the loss for mm/pm
    # with the first two dataset names ("R", "O") and the log silently lies.
    _all_obs_names = ds.obs_names.tolist() if ds.obs_names is not None else None
    if _all_obs_names is not None and cfg.obs_idx is not None:
        mech_names = [_all_obs_names[int(i)] for i in cfg.obs_idx]
    else:
        mech_names = _all_obs_names

    print(f"Data: N={N} | train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")
    print(f"Dims: P_obs={P_obs} | scaffold={cfg.scaffold} | U={U}")
    if mech_names is not None:
        print("Species:", ", ".join(str(x) for x in mech_names))

    best_val = float("inf")
    best_state = None

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_species_losses: list[np.ndarray] = []

    def _save_ckpt(path: Path, epoch: int, tag: str):
        torch.save(
            {
                "epoch": int(epoch),
                "tag": str(tag),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "opt_state": opt.state_dict(),
                "best_val": float(best_val),
                "cfg": cfg.__dict__,
            },
            path,
        )

    # When the scaffold is wider than the dataset, observations get lifted into
    # scaffold space at scaffold.obs_state_idx; obs_idx must point there so the
    # loss-time `pred[..., obs_idx]` / `y_seq[..., obs_idx]` picks the right cols.
    _lift_partial = scaffold.P != P_obs and getattr(scaffold, "obs_state_idx", None) is not None
    if _lift_partial:
        # Source positions (where mm/pm live in the dataset) — use cfg.obs_idx
        # if set, otherwise default to scaffold's obs_state_idx assuming the
        # dataset uses the same layout.
        _dataset_obs_idx = (
            list(cfg.obs_idx) if cfg.obs_idx is not None
            else list(scaffold.obs_state_idx)
        )
        if len(_dataset_obs_idx) != len(scaffold.obs_state_idx):
            raise ValueError(
                f"cfg.obs_idx (len={len(_dataset_obs_idx)}) must have the same "
                f"length as scaffold.obs_state_idx (len={len(scaffold.obs_state_idx)})."
            )
        obs_idx = torch.tensor(scaffold.obs_state_idx, device=device, dtype=torch.long)
        print(f"Partial-observability lift active: dataset_obs_idx={_dataset_obs_idx} "
              f"→ scaffold_obs_idx={scaffold.obs_state_idx}.")
    elif cfg.obs_idx is not None:
        obs_idx = torch.tensor(cfg.obs_idx, device=device, dtype=torch.long)
        print(f"Supervising only species indices: {cfg.obs_idx}")
    else:
        obs_idx = torch.arange(P_obs, device=device, dtype=torch.long)
    if not _lift_partial:
        _dataset_obs_idx = None  # unused on the non-lift code path

    # By-name IC copy data for the lift. ds.obs_names is the dataset's column
    # naming (e.g. ['R','O','m','mm','p','pm','DNA']). scaffold.state_names is
    # the scaffold's state ordering. Together they let _lift_to_scaffold_state
    # carry resource ICs (R, O) into scaffolds that need them (M5/M7/M9), with
    # _LIFT_NAME_ALIASES handling renames like O ↔ O2.
    _lift_ds_names = ([str(n) for n in ds.obs_names]
                      if _lift_partial and getattr(ds, "obs_names", None) is not None else None)
    _lift_scaf_names = (list(scaffold.state_names)
                        if _lift_partial and getattr(scaffold, "state_names", None) is not None else None)
    if _lift_partial and _lift_ds_names is not None and _lift_scaf_names is not None:
        _scaf_obs_set = set(int(i) for i in scaffold.obs_state_idx)
        _ic_copies = []
        _ds_idx_map = {n: i for i, n in enumerate(_lift_ds_names)}
        for s_idx, s_name in enumerate(_lift_scaf_names):
            if s_idx in _scaf_obs_set:
                continue
            ds_name = _LIFT_NAME_ALIASES.get(str(s_name), str(s_name))
            if ds_name in _ds_idx_map:
                _ic_copies.append(f"{ds_name}→{s_name}@{s_idx}")
        if _ic_copies:
            print(f"Lift by-name IC copy: {', '.join(_ic_copies)}")
        else:
            print("Lift by-name IC copy: no matches "
                  f"(ds_names={_lift_ds_names}, scaffold_names={_lift_scaf_names})")

    # ── L_oxygen prep: resolve u_open column + P_fluor channel once ────────
    _ox_u_col: int | None = None
    _ox_pred_ch: int | None = None
    _ox_target_ch: int | None = None
    if float(cfg.lambda_oxygen) > 0.0:
        _cn_raw = getattr(ds, "control_names", None)
        ctrl_names = [str(c) for c in _cn_raw] if _cn_raw is not None else []
        if cfg.oxygen_u_open_name in ctrl_names:
            _ox_u_col = ctrl_names.index(cfg.oxygen_u_open_name)
        else:
            print(f"[L_oxygen] WARNING: '{cfg.oxygen_u_open_name}' not in control_names "
                  f"{ctrl_names}; oxygen loss disabled.")
        # Map raw dataset channel (e.g. pm @ idx 5) into the supervised obs slice
        # so it lines up with `pred` / `y_seq` after the obs_idx selection below.
        obs_idx_list_init = (list(cfg.obs_idx) if cfg.obs_idx is not None
                             else (list(scaffold.obs_state_idx) if _lift_partial
                                   else list(range(P_obs))))
        if int(cfg.oxygen_protein_channel) in obs_idx_list_init:
            _ox_target_ch = obs_idx_list_init.index(int(cfg.oxygen_protein_channel))
            # For the lift case, scaffold.obs_state_idx[k] corresponds to the same
            # k-th position in the post-`pred = pred[:,:,obs_idx]` slice.
            _ox_pred_ch = _ox_target_ch
        else:
            print(f"[L_oxygen] WARNING: oxygen_protein_channel={cfg.oxygen_protein_channel} "
                  f"not in obs_idx={obs_idx_list_init}; oxygen loss disabled.")
            _ox_u_col = None
        if _ox_u_col is not None:
            print(f"[L_oxygen] active: lambda={cfg.lambda_oxygen} "
                  f"u_open_col={_ox_u_col} pred_ch={_ox_pred_ch} "
                  f"target_ch={_ox_target_ch}")

    # dt is now per-sample, carried by the dataloader.
    grouped_model = cfg.model_class == "ode_transformer_grouped"

    def _inject_feat_transforms(mk: dict) -> dict:
        # Only add when set so models whose forward() doesn't accept these kwargs
        # (transformer/mamba forward signatures vary) aren't disturbed.
        if cfg.u_transform and cfg.u_transform != "none":
            mk["u_transform"] = cfg.u_transform
        if cfg.y_transform and cfg.y_transform != "none":
            mk["y_transform"] = cfg.y_transform
        return mk

    for ep in range(1, cfg.epochs + 1):
        ep_t0 = time.time()
        teacher_forcing = bool(cfg.teacher_forcing) and (ep < int(cfg.tf_drop_epoch))

        # ---- train
        model.train()
        tr_total = 0.0
        tr_batches = 0
        tr_skipped_nonfinite = 0

        for y0, u_seq, y_seq, batch_lengths, z_expr_batch, dt_seq, source_idx_batch in train_loader:
            y0 = y0.to(device)
            y_seq = y_seq.to(device)
            u_seq = u_seq.to(device)
            dt_seq = dt_seq.to(device)
            if batch_lengths is not None:
                batch_lengths = batch_lengths.to(device)

            # "Gate the observation": per-batch min-subtract on y0/y_seq. No-op
            # when the flag is off, so the same .npz file can be used in both modes.
            if cfg.subtract_channel_min:
                y0, y_seq = _apply_channel_min_gate(y0, y_seq, cfg.subtract_channel_min_cols, batch_lengths)
            if _lift_partial:
                y0, y_seq = _lift_to_scaffold_state(y0, y_seq, _dataset_obs_idx, scaffold.obs_state_idx, scaffold.P, _lift_ds_names, _lift_scaf_names)

            opt.zero_grad(set_to_none=True)

            # Per-sample loss weights from source_loss_weights config. Map each
            # sample's source_idx (0=old, 1=new, 2=synth) to its multiplier.
            # Applied only in training (not val/test) so eval metrics stay raw.
            _sample_w = None
            if cfg.source_loss_weights:
                _idx_to_name = {0: "old", 1: "new", 2: "synth"}
                _name_to_w = {k: float(v) for k, v in cfg.source_loss_weights.items()}
                _sample_w = torch.tensor(
                    [_name_to_w.get(_idx_to_name.get(int(s), ""), 1.0) for s in source_idx_batch.tolist()],
                    device=device, dtype=torch.float32,
                )

            # Per-sample per-channel mask: zero specified channels for synth
            # (source==2) samples so no-expression trajectories don't bias the
            # model toward predicting zero protein on real expressing samples.
            # Only built when synth_obs_mask_channels is configured AND the batch
            # actually contains synth samples — otherwise stays None (no-op).
            _synth_ch_mask = None
            _synth_ep_keep = None   # (B,) bool for endpoint_mse exclusion
            if cfg.synth_obs_mask_channels and source_idx_batch is not None:
                is_synth = (source_idx_batch == 2)  # (B,) on CPU
                if is_synth.any():
                    obs_idx_list_m = (list(cfg.obs_idx) if cfg.obs_idx is not None
                                      else (obs_idx.tolist() if torch.is_tensor(obs_idx)
                                            else list(obs_idx)))
                    P_m = len(obs_idx_list_m)
                    ch_mask = torch.ones(len(source_idx_batch), P_m,
                                         device=device, dtype=torch.float32)
                    for raw_ch in cfg.synth_obs_mask_channels:
                        if raw_ch in obs_idx_list_m:
                            local_ch = obs_idx_list_m.index(raw_ch)
                            ch_mask[is_synth.to(device), local_ch] = 0.0
                    _synth_ch_mask = ch_mask
                    # For endpoint_mse: exclude synth rows when the endpoint
                    # channel is among the masked channels.
                    if cfg.endpoint_channels and any(
                        int(c) in cfg.synth_obs_mask_channels
                        for c in cfg.endpoint_channels
                    ):
                        _synth_ep_keep = (~is_synth).to(device)

            model_kwargs = {
                "teacher_forcing": teacher_forcing,
                "tf_every": int(cfg.tf_every),
            }
            _inject_feat_transforms(model_kwargs)
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
            if cfg.use_ivtt_mse_loss:
                loss = loss_fn_ivtt_mse(pred_full=pred, theta=theta, y_seq_full=y_seq, lengths=batch_lengths)
                # keep these for any logging/diagnostics below
                pred = pred[:, :, obs_idx]
                y_seq = y_seq[:, :, obs_idx]
            else:
                pred = pred[:, :, obs_idx]
                y_seq = y_seq[:, :, obs_idx]

                if cfg.supervise_endpoints_only:
                    pred_l  = torch.stack([pred[:, 0, :],  pred[:, -1, :]],  dim=1)
                    y_seq_l = torch.stack([y_seq[:, 0, :], y_seq[:, -1, :]], dim=1)
                    loss = loss_fn(pred_l, y_seq_l, None, use_log_loss=use_log_loss,
                                   sample_weights=_sample_w)
                else:
                    # obs_idx is a torch.Tensor here; channel-keyed config knobs
                    # use raw state indices, so resolve against the python list.
                    # Use cfg.obs_idx (user-specified raw state indices) when present —
                    # the partial-observability lift may have remapped `obs_idx` to
                    # scaffold-local indices, but channel-keyed config dicts (species_weights,
                    # time_upweight, endpoint_channels) reference raw indices.
                    obs_idx_list = (list(cfg.obs_idx) if cfg.obs_idx is not None
                                    else (obs_idx.tolist() if torch.is_tensor(obs_idx) else list(obs_idx)))
                    chan_w = _resolve_channel_weights(obs_idx_list, cfg.species_weights, pred.device, pred.dtype)
                    time_w = _build_per_channel_time_weight(
                        obs_idx_list, cfg.time_upweight, batch_lengths,
                        pred.shape[0], pred.shape[1], pred.device, pred.dtype,
                    )
                    loss = loss_fn(pred, y_seq, batch_lengths,
                                   use_log_loss=use_log_loss,
                                   channel_weights=chan_w,
                                   time_weight=time_w,
                                   clamp_min=float(cfg.loss_clamp_min),
                                   sample_weights=_sample_w,
                                   sample_channel_mask=_synth_ch_mask)
                    if cfg.lambda_endpoint > 0.0 and cfg.endpoint_channels:
                        ep_post = [obs_idx_list.index(int(c)) for c in cfg.endpoint_channels]
                        loss = loss + float(cfg.lambda_endpoint) * endpoint_mse(
                            pred, y_seq, batch_lengths, ep_post,
                            use_log_loss=use_log_loss,
                            clamp_min=float(cfg.loss_clamp_min),
                            sample_mask=_synth_ep_keep,
                            loss_space=str(cfg.endpoint_loss_space))
                    if cfg.lambda_peak > 0.0 and cfg.peak_channels:
                        pk_post = [obs_idx_list.index(int(c)) for c in cfg.peak_channels]
                        loss = loss + float(cfg.lambda_peak) * peak_mse(
                            pred, y_seq, batch_lengths, pk_post,
                            use_log_loss=use_log_loss,
                            clamp_min=float(cfg.loss_clamp_min),
                            sample_mask=_synth_ep_keep,
                            loss_space=str(cfg.endpoint_loss_space))
                    if cfg.loss_normalizer_channels:
                        loss = loss / float(cfg.loss_normalizer_channels)

                    # ── L_oxygen (tex eq.) ────────────────────────────────
                    # Post-tube-opening MSE on the fluorescent-protein channel.
                    # Applied only to samples whose `u_open` column fires within
                    # their valid length. Predictions/targets are taken in raw
                    # (linear) space; the same log1p+clamp_min treatment used by
                    # endpoint_mse is applied for scale parity with L_traj.
                    if (float(cfg.lambda_oxygen) > 0.0 and _ox_u_col is not None
                            and _ox_pred_ch is not None):
                        B_o, K_o = pred.shape[0], pred.shape[1]
                        u_open_seq = u_seq[:, :K_o, _ox_u_col]  # (B,K)
                        if batch_lengths is not None:
                            valid_mask = _build_loss_mask(batch_lengths, K_o, device)
                            u_open_seq = u_open_seq * valid_mask.to(u_open_seq.dtype)
                        has_event = (u_open_seq > 0).any(dim=1)  # (B,)
                        if has_event.any():
                            t_open = torch.argmax((u_open_seq > 0).to(torch.long), dim=1)  # (B,)
                            t_grid = torch.arange(K_o, device=device).unsqueeze(0)        # (1,K)
                            post_mask = (t_grid > t_open.unsqueeze(1))                    # (B,K)
                            if batch_lengths is not None:
                                post_mask = post_mask & _build_loss_mask(batch_lengths, K_o, device)
                            post_mask = post_mask & has_event.unsqueeze(1)
                            p_ox = pred[..., _ox_pred_ch]
                            y_ox = y_seq[..., _ox_pred_ch]
                            if use_log_loss:
                                clamp_v = float(cfg.loss_clamp_min)
                                p_ox = torch.log1p(p_ox.clamp_min(clamp_v))
                                y_ox = torch.log1p(y_ox.clamp_min(clamp_v))
                            sq = (p_ox - y_ox).pow(2) * post_mask.to(p_ox.dtype)
                            denom = post_mask.sum().clamp_min(1).to(p_ox.dtype)
                            ox_loss = sq.sum() / denom
                            loss = loss + float(cfg.lambda_oxygen) * ox_loss

            # Model 7 zero-trajectory boundary loss for synthetic no-go samples
            # (z_expr == 0). Pushes pred (mm, pm) toward zero on those samples
            # so the encoder/scaffold learns the infeasible-expression region.
            # No-op when lambda_zero_traj=0 (default) or no z=0 samples in batch.
            if float(cfg.lambda_zero_traj) > 0.0:
                z_batch = z_expr_batch.to(device)
                no_go_mask = (z_batch == 0)
                if no_go_mask.any():
                    no_go_pred = pred[no_go_mask]
                    # Apply the same log1p scale as the trajectory loss when
                    # use_log_loss is on. log1p(0)=0 so the "push pred → 0"
                    # target is still 0; this just normalises the magnitudes so
                    # lambda_zero_traj is comparable to the trajectory MSE.
                    # Without this, raw-space (75² = 5625) swamps log1p-space
                    # trajectory MSE (~0.1) by 250× per sample.
                    if use_log_loss:
                        no_go_pred = torch.log1p(no_go_pred.clamp_min(0.0))
                    if batch_lengths is not None:
                        ng_lengths = batch_lengths[no_go_mask]
                        K_b = no_go_pred.shape[1]
                        time_mask = _build_loss_mask(ng_lengths, K_b, device)
                        sq = no_go_pred.pow(2) * time_mask.unsqueeze(-1)
                        denom = time_mask.sum().clamp_min(1) * no_go_pred.shape[-1]
                        zero_loss = sq.sum() / denom
                    else:
                        zero_loss = no_go_pred.pow(2).mean()
                    loss = loss + float(cfg.lambda_zero_traj) * zero_loss

            if cfg.l1_regularization:
                reg_loss = torch.mean(torch.abs(theta[:,1:,:] - theta[:,:-1,:]))
                loss += cfg.lambda_reg * reg_loss

            if cfg.l2_regularization:
                reg_loss = torch.mean((theta[:,1:,:] - theta[:,:-1,:]).pow(2))
                loss += cfg.lambda_reg * reg_loss

            loss.backward()

            # Always compute the global grad norm so the step can be gated on its
            # finiteness. max_norm=inf is a no-op clip (it still returns the norm),
            # so the configured clipping behaviour is preserved when grad_clip is set.
            _max_norm = float(cfg.grad_clip) if (cfg.grad_clip and float(cfg.grad_clip) > 0) else float("inf")
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), _max_norm)

            # A single batch can produce non-finite grads (fp32 overflow in the long
            # BPTT rollout, or 0*inf in a scaffold). clip_grad_norm_ PROPAGATES NaN/Inf
            # rather than removing it, so stepping here would poison every weight with
            # NaN and kill the whole run. Skip the update for this batch instead.
            if not torch.isfinite(total_norm):
                opt.zero_grad(set_to_none=True)
                tr_skipped_nonfinite += 1
                continue

            opt.step()

            tr_total += float(loss.item())
            tr_batches += 1

        tr_loss = tr_total / max(1, tr_batches)
        train_losses.append(tr_loss)
        if tr_skipped_nonfinite > 0:
            print(f"  [grad guard] skipped {tr_skipped_nonfinite} batch(es) with non-finite "
                  f"gradients ({tr_batches} applied)")

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

            with torch.no_grad():
                for y0, u_seq, y_seq, batch_lengths, z_expr_batch, dt_seq, source_idx_batch in val_loader:
                    y0 = y0.to(device)
                    y_seq = y_seq.to(device)
                    u_seq = u_seq.to(device)
                    dt_seq = dt_seq.to(device)
                    if batch_lengths is not None:
                        batch_lengths = batch_lengths.to(device)

                    if cfg.subtract_channel_min:
                        y0, y_seq = _apply_channel_min_gate(y0, y_seq, cfg.subtract_channel_min_cols, batch_lengths)
                    if _lift_partial:
                        y0, y_seq = _lift_to_scaffold_state(y0, y_seq, _dataset_obs_idx, scaffold.obs_state_idx, scaffold.P, _lift_ds_names, _lift_scaf_names)

                    if cfg.use_ivtt_mse_loss:
                        model_kwargs = {"teacher_forcing": False, "tf_every": int(cfg.tf_every)}
                        _inject_feat_transforms(model_kwargs)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.autocast_bf16 and device.type == "cuda")):
                            pred_full, theta, _ = model(y0, u_seq, dt_seq, obs_idx, y_seq, **model_kwargs)
                        loss = loss_fn_ivtt_mse(pred_full=pred_full, theta=theta, y_seq_full=y_seq, lengths=batch_lengths)
                        va_total += float(loss.item())
                        va_batches += 1
                        continue

                    model_kwargs = {"y_seq": None, "teacher_forcing": False}
                    _inject_feat_transforms(model_kwargs)
                    if grouped_model and batch_lengths is not None:
                        model_kwargs["lengths"] = batch_lengths
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.autocast_bf16 and device.type == "cuda")):
                        pred, _, _ = model(y0, u_seq, dt_seq, obs_idx, **model_kwargs)
                    pred = pred[:, :, obs_idx]
                    y_seq = y_seq[:, :, obs_idx]
                    if cfg.supervise_endpoints_only:
                        pred_l  = torch.stack([pred[:, 0, :],  pred[:, -1, :]],  dim=1)
                        y_seq_l = torch.stack([y_seq[:, 0, :], y_seq[:, -1, :]], dim=1)
                        loss = loss_fn(pred_l, y_seq_l, None, use_log_loss=use_log_loss)
                    else:
                        # Use cfg.obs_idx (user-specified raw state indices) when present —
                        # the partial-observability lift may have remapped `obs_idx` to
                        # scaffold-local indices, but channel-keyed config dicts (species_weights,
                        # time_upweight, endpoint_channels) reference raw indices.
                        obs_idx_list = (list(cfg.obs_idx) if cfg.obs_idx is not None
                                        else (obs_idx.tolist() if torch.is_tensor(obs_idx) else list(obs_idx)))
                        chan_w = _resolve_channel_weights(obs_idx_list, cfg.species_weights, pred.device, pred.dtype)
                        time_w = _build_per_channel_time_weight(
                            obs_idx_list, cfg.time_upweight, batch_lengths,
                            pred.shape[0], pred.shape[1], pred.device, pred.dtype,
                        )
                        loss = loss_fn(pred, y_seq, batch_lengths,
                                       use_log_loss=use_log_loss,
                                       channel_weights=chan_w,
                                       time_weight=time_w,
                                       clamp_min=float(cfg.loss_clamp_min))
                        if cfg.lambda_endpoint > 0.0 and cfg.endpoint_channels:
                            ep_post = [obs_idx_list.index(int(c)) for c in cfg.endpoint_channels]
                            loss = loss + float(cfg.lambda_endpoint) * endpoint_mse(
                                pred, y_seq, batch_lengths, ep_post,
                                use_log_loss=use_log_loss,
                                clamp_min=float(cfg.loss_clamp_min),
                                loss_space=str(cfg.endpoint_loss_space))
                        if cfg.lambda_peak > 0.0 and cfg.peak_channels:
                            pk_post = [obs_idx_list.index(int(c)) for c in cfg.peak_channels]
                            loss = loss + float(cfg.lambda_peak) * peak_mse(
                                pred, y_seq, batch_lengths, pk_post,
                                use_log_loss=use_log_loss,
                                clamp_min=float(cfg.loss_clamp_min),
                                loss_space=str(cfg.endpoint_loss_space))
                        if cfg.loss_normalizer_channels:
                            loss = loss / float(cfg.loss_normalizer_channels)
                        if (float(cfg.lambda_oxygen) > 0.0 and _ox_u_col is not None
                                and _ox_pred_ch is not None):
                            B_o, K_o = pred.shape[0], pred.shape[1]
                            u_open_seq = u_seq[:, :K_o, _ox_u_col]
                            if batch_lengths is not None:
                                valid_mask = _build_loss_mask(batch_lengths, K_o, device)
                                u_open_seq = u_open_seq * valid_mask.to(u_open_seq.dtype)
                            has_event = (u_open_seq > 0).any(dim=1)
                            if has_event.any():
                                t_open = torch.argmax((u_open_seq > 0).to(torch.long), dim=1)
                                t_grid = torch.arange(K_o, device=device).unsqueeze(0)
                                post_mask = (t_grid > t_open.unsqueeze(1))
                                if batch_lengths is not None:
                                    post_mask = post_mask & _build_loss_mask(batch_lengths, K_o, device)
                                post_mask = post_mask & has_event.unsqueeze(1)
                                p_ox = pred[..., _ox_pred_ch]
                                y_ox = y_seq[..., _ox_pred_ch]
                                if use_log_loss:
                                    clamp_v = float(cfg.loss_clamp_min)
                                    p_ox = torch.log1p(p_ox.clamp_min(clamp_v))
                                    y_ox = torch.log1p(y_ox.clamp_min(clamp_v))
                                sq = (p_ox - y_ox).pow(2) * post_mask.to(p_ox.dtype)
                                denom = post_mask.sum().clamp_min(1).to(p_ox.dtype)
                                loss = loss + float(cfg.lambda_oxygen) * (sq.sum() / denom)
                    va_total += float(loss.item())

                    sp = loss_fn_per_species(pred, y_seq, batch_lengths, use_log_loss=use_log_loss).detach().cpu()
                    sp_total = sp if sp_total is None else sp_total + sp
                    va_batches += 1

            va_loss = va_total / max(1, va_batches)
            val_losses.append(va_loss)

            if sp_total is not None:
                sp_last = (sp_total / max(1, va_batches)).numpy()
                val_species_losses.append(sp_last)

            if va_loss < best_val:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        ep_time = time.time() - ep_t0

        if va_loss is None:
            print(f"ep {ep:4d} | train {tr_loss:.6f} | tf={int(teacher_forcing)} | {ep_time:.2f}s")
        else:
            sp_str = ""
            if sp_last is not None:
                if mech_names is None:
                    sp_str = "  [" + "  ".join(f"{v:.4f}" for v in sp_last) + "]"
                else:
                    sp_str = "  [" + "  ".join(f"{n}:{v:.4f}" for n, v in zip(mech_names, sp_last)) + "]"
            print(
                f"ep {ep:4d} | train {tr_loss:.6f} | val {va_loss:.6f} | best {best_val:.6f} | tf={int(teacher_forcing)}{sp_str} | {ep_time:.2f}s"
            )

        if math.isnan(tr_loss) or math.isnan(va_loss):
            print(f"NaN detected at epoch {ep} — stopping early.")
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
            for y0, u_seq, y_seq, batch_lengths, z_expr_batch, dt_seq, source_idx_batch in test_loader:
                y0 = y0.to(device)
                y_seq = y_seq.to(device)
                u_seq = u_seq.to(device)
                dt_seq = dt_seq.to(device)
                if batch_lengths is not None:
                    batch_lengths = batch_lengths.to(device)
                if cfg.subtract_channel_min:
                    y0, y_seq = _apply_channel_min_gate(y0, y_seq, cfg.subtract_channel_min_cols, batch_lengths)
                if _lift_partial:
                    y0, y_seq = _lift_to_scaffold_state(y0, y_seq, _dataset_obs_idx, scaffold.obs_state_idx, scaffold.P, _lift_ds_names, _lift_scaf_names)
                model_kwargs = {"y_seq": None, "teacher_forcing": False}
                _inject_feat_transforms(model_kwargs)
                if grouped_model and batch_lengths is not None:
                    model_kwargs["lengths"] = batch_lengths
                pred, _, _ = model(y0, u_seq, dt_seq, obs_idx, **model_kwargs)
                pred = pred[:, :, obs_idx]
                y_seq = y_seq[:, :, obs_idx]
                loss = loss_fn(pred, y_seq, batch_lengths, use_log_loss=use_log_loss,
                               clamp_min=float(cfg.loss_clamp_min))
                te_total += float(loss.item())
                sp = loss_fn_per_species(pred, y_seq, batch_lengths, use_log_loss=use_log_loss).detach().cpu()
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
        {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
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

            plot_experiment(exp_dir, n_samples=int(plot_samples), sample_idx=int(plot_sample_idx),
                            extra_new_samples=int(getattr(cfg, "plot_extra_new_samples", 0)))

            # epochs=None => automatically uses available checkpoints and picks up to max_overlays evenly spaced.
            # Needs per-epoch ckpt_ep*.pt files; when ckpt_every=0 none exist, so skip cleanly.
            if int(getattr(cfg, "ckpt_every", 0)) > 0:
                plot_epoch_prediction_overlays(
                    exp_dir,
                    sample_idx=int(plot_sample_idx),
                    epochs=None,
                    max_overlays=8,
                )
            else:
                print("[plot] ckpt_every=0 → no per-epoch checkpoints; skipping epoch-overlay plots.")
        except ImportError as e:
            import sys
            print(f"[plot] plot_diagnostics.py import failed: {e}; sys.path={sys.path}; skipping plots.")
        except Exception as e:
            import traceback
            print(f"[plot] failed: {type(e).__name__}: {e}")
            traceback.print_exc()

    if cfg.endpoint_r2:
        try:
            from metrics import endpoint_r2
            from plot_diagnostics import device_auto as _dev_auto

            r2_device = _dev_auto()
            print("\nRunning endpoint R² analysis...")
            result = endpoint_r2.collect_endpoints(
                exp_dir, r2_device, split="test", protein_sp="pm", mrna_sp="mm"
            )
            r2_protein = endpoint_r2.r2(result["true_protein_final"], result["pred_protein_final"])
            r2_mrna    = endpoint_r2.r2(result["true_mrna_max"],      result["pred_mrna_max"])
            print(f"  R²(protein final) = {r2_protein:.4f}")
            print(f"  R²(mRNA max)      = {r2_mrna:.4f}")

            # Stratified R² by data source (old=0, new=1) — analysis only,
            # does not affect training or model selection.
            src_names = {0: "old", 1: "new"}
            by_src = endpoint_r2.r2_by_source(result)
            for src_id, src_name in src_names.items():
                if src_id in by_src:
                    s = by_src[src_id]
                    print(f"  R²(protein final) [{src_name:3s}] = {s['r2_protein']:.4f}  (n={s['n']})")
                    print(f"  R²(mRNA max)      [{src_name:3s}] = {s['r2_mrna']:.4f}")

            out_path = exp_dir / "endpoint_r2.png"
            endpoint_r2.plot_endpoints(
                [result], protein_sp="pm", mrna_sp="mm", split="test", out_path=out_path
            )
            endpoint_r2.plot_endpoints_by_source(
                result, protein_sp="pm", mrna_sp="mm", split="test",
                out_path=exp_dir / "endpoint_r2_by_source.png",
            )
            endpoint_r2.save_r2_cache(exp_dir, result, r2_protein, r2_mrna)

            if wandb_run is not None:
                wandb_run.summary["endpoint_r2/protein_final"] = float(r2_protein)
                wandb_run.summary["endpoint_r2/mrna_max"]      = float(r2_mrna)
                for src_id, src_name in src_names.items():
                    if src_id in by_src:
                        s = by_src[src_id]
                        wandb_run.summary[f"endpoint_r2/protein_{src_name}"] = s["r2_protein"]
                        wandb_run.summary[f"endpoint_r2/mrna_{src_name}"]    = s["r2_mrna"]
        except Exception as e:
            print(f"[endpoint_r2] failed: {e}")

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
