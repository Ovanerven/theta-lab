#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 11:45:35 2025

@author: bob-van-sluijs
"""

from __future__ import annotations
from utils import *
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import os
import torch
import copy
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from typing import Mapping, Hashable, Sequence, Dict, List, Tuple
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
from sklearn.metrics import r2_score

# -----------------------------------------------------------------------------
# 1) Vectorized Dataset + Caching
# -----------------------------------------------------------------------------
class Datastruct(Dataset):
    def __init__(
        self,
        x0_list : List[torch.Tensor],
        u_list  : List[torch.Tensor],
        dna_raw : List[torch.Tensor],   #                       
        dt_list : List[torch.Tensor],
        y_list  : List[torch.Tensor],
    ):
        self.x0_list  = x0_list
        self.u_list   = u_list
        self.dna_raw  = dna_raw
        self.dt_list  = dt_list
        self.y_list   = y_list

    def __len__(self): 
         return len(self.x0_list)

    def __getitem__(self, idx):
        return (
            self.x0_list[idx],
            self.u_list[idx],
            self.dna_raw[idx],
            self.dt_list[idx],
            self.y_list[idx],
        )


# ---------- compute μ/σ for observed channels on TRAIN only ----------
def _fit_obs_mu_std(y_list, transform="", eps=1e-8, device="cpu"):
    """
    y_list: list of (K,3) tensors with channels [mRNA(mm), p(unused), protein(pm)]
    train_idx: iterable of indices to include (training split)
    Returns four Python floats: mu_mm, std_mm, mu_pm, std_pm (on transformed data)
    """
    mm_vals, pm_vals = [], []
    for i in range(len(y_list)):
        y = y_list[i].to(device)      # (K,3)
        mm = y[:, 0].clamp_min(0.0)   # mRNA (observed)
        pm = y[:, 2].clamp_min(0.0)   # mature protein (observed)
        mm_vals.append(mm)
        pm_vals.append(pm)

    mm_all = torch.cat(mm_vals, dim=0)  # (N,)
    pm_all = torch.cat(pm_vals, dim=0)  # (N,)

    # variance-stabilizing transform
    if transform == "sqrt":
        mm_t = torch.sqrt(mm_all)
        pm_t = torch.sqrt(pm_all)
    elif transform == "log1p":
        mm_t = torch.log1p(mm_all)
        pm_t = torch.log1p(pm_all)
    elif transform == "anscombe":
        mm_t = 2.0 * torch.sqrt(mm_all + 0.375)
        pm_t = 2.0 * torch.sqrt(pm_all + 0.375)
    else:
        mm_t, pm_t = mm_all, pm_all

    # drop non-finite just in case
    mm_t = mm_t[torch.isfinite(mm_t)]
    pm_t = pm_t[torch.isfinite(pm_t)]

    mu_mm  = float(mm_t.mean())
    std_mm = float(mm_t.std(unbiased=False).clamp_min(eps))
    mu_pm  = float(pm_t.mean())
    std_pm = float(pm_t.std(unbiased=False).clamp_min(eps))
    return mu_mm, std_mm, mu_pm, std_pm

def smooth_gaussian_preserve_total(
    u_seq: torch.Tensor,                  # (B, K, U)
    sigma: float = 3.0,                   # std in steps
    radius: int | None = None,            # kernel radius; default ~ 3σ
    lengths: torch.Tensor | None = None,  # (B,) valid lengths; optional
) -> torch.Tensor:
    """
    Symmetric Gaussian smoothing in time (left+right neighbors), per-channel,
    preserving *per-(batch,channel)* total volume over valid steps.

    Returns: smoothed tensor of shape (B, K, U).
    """
    B, K, U = u_seq.shape
    device, dtype = u_seq.device, u_seq.dtype
    if radius is None:
        radius = max(1, int(round(3.0 * sigma)))  # ~99.7% mass

    # 1) Build normalized symmetric Gaussian kernel
    t = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / (k.sum() + 1e-12)                      # sum = 1
    weight = k.view(1, 1, -1).repeat(U, 1, 1)      # (U,1,L) for grouped conv

    # 2) Prepare data as (B, U, K); optionally extend tail to last valid frame
    x = u_seq.transpose(1, 2).contiguous()         # (B, U, K)

    if lengths is not None:
        # make a mask over time for valid steps
        tgrid = torch.arange(K, device=device).unsqueeze(0)              # (1,K)
        valid_mask = (tgrid < lengths.view(-1, 1)).unsqueeze(1).to(dtype)  # (B,1,K)

        # repeat last valid frame across the padded tail so blur doesn't see zeros
        for b in range(B):
            L = int(lengths[b].item())
            if L < 1: continue
            if L < K:
                last = x[b, :, L-1:L]                                    # (U,1)
                x[b, :, L:] = last                                       # extend
    else:
        valid_mask = torch.ones(B, 1, K, device=device, dtype=dtype)

    # 3) Symmetric blur with reflect padding (no future leakage beyond K)
    xpad = F.pad(x, (radius, radius), mode='reflect')   # (B,U,K+2r)
    y = F.conv1d(xpad, weight, padding=0, groups=U)     # (B,U,K), per-channel

    # 4) EXACT total preservation per (B,U) over valid steps
    s_orig   = (x * valid_mask).sum(dim=2)              # (B,U)
    s_smooth = (y * valid_mask).sum(dim=2) + 1e-12      # (B,U)
    scale = (s_orig / s_smooth).unsqueeze(2)            # (B,U,1)
    y = y * scale

    # 5) Zero-out padded region (if any) to keep shapes consistent
    y = y * valid_mask + 0.0 * (1.0 - valid_mask)

    return y.transpose(1, 2).contiguous()               # (B,K,U)

# -----------------------------------------------------------------------------
# 1) Model/ML infrastructure
# -----------------------------------------------------------------------------
from pathlib import Path
import random, numpy as np, torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.preprocessing import StandardScaler

class NeuralSpline:
    def __init__(
        self,
        path: str | Path = None,
        inputs=None,
        outputs=None,
        time_label: str = "Time_seconds",
        varlist=('DNA c',),
        rescale_inputs: bool = True, w = 1,
    ):
        
        # ---------- load  ------------------------------------------------
        if inputs is None or outputs is None:
            from parse_IVTT_data import load_parsed_io
            self.inputs_df, self.outputs_df, self.metadata_dict = \
                load_parsed_io(f"/home/bob-van-sluijs/Desktop/{path}/")
        else:
            self.inputs_df, self.outputs_df = inputs, output
        self.w = w

        # ---------- diffs & intervals per run ---------------------------
        input_list, interval_list, dna_raw_list = [], [], []
        for tag, df in self.inputs_df.items():
            cols = sorted([c for c in df.columns if c != time_label])
            diffs_raw = df[cols].to_numpy()[0:-1]          # (K,U)
            interval_list.append(list(zip(df[time_label][:-1],
                                          df[time_label][0:])))
            dna_cum = np.cumsum(diffs_raw[:, [cols.index('DNA c')]])
            """
            # dna_cum_final = float(dna_cum[-1])
            # dna_conc = []
            # for i in range(len(dna_cum)):
            #     if dna_cum[i] == 0:
            #         dna_conc.append(dna_cum_final)
            #     else:
            #         dna_conc.append(dna_cum_final)
            """
            dna_raw_list.append(diffs_raw[:, [cols.index('DNA c')]])

            diffs_raw = np.delete(diffs_raw, cols.index('DNA c'), 1)    
            input_list.append(diffs_raw)

        
        # ---------- optional log‑standard‑scale (excluding DNA) ----------        
        rescale_inputs = rescale_inputs
        if rescale_inputs:
            all_arr      = np.concatenate(input_list, 0)
            self.scaler       = MinMaxScaler().fit(all_arr)
            for i, arr in enumerate(input_list):
                input_list[i] = self.scaler.transform(arr)
                print('rescale',i, input_list[i] , arr)
 
        # ---------- build tensor lists (3‑state) -------------------------
        self.x0_list, self.u_list, self.dna_list, self.dt_list, self.y_list = \
            [], [], [], [], []

        self.input_cols = sorted({c for d in self.inputs_df.values()
                                  for c in d.columns if c != time_label})
        self.outputs_list = [
            self.outputs_df[tag].sort_values(time_label).reset_index(drop=True)
            for tag in self.inputs_df.keys()
            ]
        
        for (intervals, diffs_scaled), dna_raw, df_out in zip(
                zip(interval_list, input_list), dna_raw_list, self.outputs_list):

            times = df_out[time_label].to_numpy()
            bro   = df_out["Broccoli [RFU]"].to_numpy() # mRNA
            mch   = df_out["mCherry [RFU]"].to_numpy()/2    # mature protein

            dt_seq  = torch.tensor(times[1:] - times[:-1], dtype=torch.float32)
            u_seq   = torch.tensor(diffs_scaled,          dtype=torch.float32)
            dna_seq = torch.tensor(dna_raw,               dtype=torch.float32)
            self.dna_fixed  = dna_raw
            y_arr = np.stack([bro[:-1],
                              np.zeros_like(bro[:-1]),      # p unmeasured
                              mch[:-1]], axis=1)
            y_seq = torch.tensor(y_arr, dtype=torch.float32)
            
            if mch[-1] < 7000:
                self.x0_list.append(y_seq[0])   # (3,)
                self.u_list.append(u_seq)       # (K,U‑DNA)
                self.dna_list.append(dna_seq)   # (K,1)
                self.dt_list.append(dt_seq)     # (K,)
                self.y_list.append(y_seq)       # (K,3)
                
 
        
        self.ctrl_idx = [self.input_cols.index(v) for v in varlist]
        self.device   = torch.device("cuda")

        self.make_train_test_split()
        self.compute_channel_weight()
            
    
    def MC_sample_U(
        self,
        time_col: str = "Time_seconds",
        n_per_original: int = 100,
        # where additions are allowed; None means use grid start/end
        time_window: Tuple[Optional[float], Optional[float]] = (None, None),
        # experiment duration: None -> use largest final time across all inputs
        duration_s: Optional[float] = None,
        # max number of bolus events per column (int for all cols, or per-col dict)
        max_events_per_col: Union[int, Dict[str, int]] = 3,
        # minimum time separation between two boluses in *the same column* (seconds)
        min_separation_s: float = 30.0,
        # compute bounds from data then stretch by 50% (i.e., [0.5*min, 1.5*max], clamped ≥0)
        stretch: float = 0.5,
        # manual {col: (min, max)} overrides computed bounds
        manual_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        seed: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        """
        Sample new U tables with the constraint:
          - per 60 s row → at most one nonzero addition and from exactly one species.
          - per species → at most `max_events_per_col` events (default 3).
          - (optional) per-species minimum separation in time.
    
        Bounds are computed once across all inputs, then used for all samples.
        """
        import math, random
        import numpy as np
        import pandas as pd
    
        rng = random.Random(seed)
    
        # ---------- 0) Gather source inputs ----------
        U_list = list(self.inputs_df.values())
        if len(U_list) == 0:
            raise ValueError("No input dataframes found in self.inputs_df.")
    
        # ---------- 1) Compute global column set and bounds ----------
        # reference column order (excluding time): keep order from the first df
        ref_df = U_list[0]
        if time_col not in ref_df.columns:
            raise ValueError(f"Time column '{time_col}' not found in the first input.")
        ref_cols = [c for c in ref_df.columns if c != time_col]
    
        # union of columns across all inputs (ex time)
        all_cols = sorted(set().union(*[set(df.columns) for df in U_list]))
        if time_col not in all_cols:
            raise ValueError(f"Time column '{time_col}' not found in inputs.")
        data_cols_all = [c for c in all_cols if c != time_col]
    
        # ensure any column missing in the reference order gets appended at the end
        cols = ref_cols + [c for c in data_cols_all if c not in ref_cols]
    
        # global min/max over provided tables, then stretch
        bounds: Dict[str, Tuple[float, float]] = {}
        for c in data_cols_all:
            col_min = float(min(df[c].min() if c in df.columns else 0.0 for df in U_list))
            col_max = float(max(df[c].max() if c in df.columns else 0.0 for df in U_list))
            lo = max(0.0, col_min * (1.0 - stretch))  # clamp lower at 0
            hi = col_max * (1.0 + stretch)
            # robust fallback
            if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
                lo, hi = 0.0, 0.0
            bounds[c] = (lo, hi)
    
        # manual overrides
        if manual_bounds:
            for k, v in manual_bounds.items():
                if k in bounds and isinstance(v, (list, tuple)) and len(v) == 2:
                    lo, hi = float(v[0]), float(v[1])
                    bounds[k] = (max(0.0, lo), max(0.0, hi))
    
        # ---------- 2) Build a global time grid (60 s spacing) ----------
        # global t0 = smallest first time, global T = largest final time (unless overridden)
        t0 = min(float(df[time_col].iloc[0]) for df in U_list)
        if duration_s is None:
            duration_s = max(float(df[time_col].iloc[-1]) for df in U_list)
    
        dt = 60.0  # exactly 60 seconds per your requirement
        n_steps = int(round((duration_s - t0) / dt))
        n_steps = max(n_steps, 1)
        t = t0 + np.arange(n_steps + 1) * dt
        K = t.size
    
        # allowed time window
        win_lo = t0 if time_window[0] is None else float(time_window[0])
        win_hi = duration_s if time_window[1] is None else float(time_window[1])
        allowed_mask = (t >= win_lo) & (t <= win_hi)
        allowed_idx_all = np.where(allowed_mask)[0].tolist()
    
        # We only place events on *rows* (time indices). Enforce "one event per row"
        # by tracking a pool of available indices and removing an index once used.
    
        # ---------- helpers ----------
        def _max_events_for(col: str) -> int:
            if isinstance(max_events_per_col, int):
                return max(0, int(max_events_per_col))
            return max(0, int(max_events_per_col.get(col, 0)))
    
        step_gap = max(1, int(round(min_separation_s / max(dt, 1e-12)))) if min_separation_s > 0 else 1
    
        def _pick_indices_from_pool(pool: List[int], n_events: int) -> List[int]:
            """Pick indices from the shared pool, respecting within-species min separation."""
            if n_events <= 0 or not pool:
                return []
            chosen: List[int] = []
            # Work on a local pool copy because we also need to enforce intra-species spacing
            cand = sorted(pool)
            while cand and len(chosen) < n_events:
                i = rng.choice(cand)
                chosen.append(i)
                # remove neighbors within step_gap for this species
                cand = [j for j in cand if abs(j - i) >= step_gap]
            # Final chosen must exist in the shared pool; caller will remove them from global pool.
            return sorted(chosen)
    
        # ---------- 3) Sample new experiments ----------
        outputs: List[pd.DataFrame] = []
    
        for _ in range(n_per_original):
            print(_)
            # fresh pool of time indices allowed for this experiment
            pool = allowed_idx_all.copy()
            rng.shuffle(pool)
    
            # zero frame
            new_df = pd.DataFrame(0.0, index=np.arange(K), columns=[time_col] + cols)
            new_df[time_col] = t
    
            # Decide per-species counts without exceeding pool size.
            # Start with max caps, then randomly reduce while sum > |pool|.
            caps = {c: _max_events_for(c) for c in cols}
            # quickly bound total events
            total_cap = sum(caps.values())
            tot_allowed = min(total_cap, len(pool))
            # draw a random total number of events in [0, tot_allowed]
            total_events = rng.randint(0, tot_allowed)
    
            # Distribute total_events across species with caps
            counts = {c: 0 for c in cols}
            available_species = [c for c in cols if caps[c] > 0]
            for _e in range(total_events):
                if not available_species or not pool:
                    break
                c = rng.choice(available_species)
                counts[c] += 1
                if counts[c] >= caps[c]:
                    available_species.remove(c)
    
            # For each species, pick its indices from the *shared* pool, removing used rows
            for c in rng.sample(cols, k=len(cols)):  # randomize species order
                n_c = counts[c]
                if n_c <= 0:
                    continue
                if not pool:
                    break
                idxs = _pick_indices_from_pool(pool, n_c)
                # ensure we don't reuse rows globally
                used = set()
                for k_idx in idxs:
                    if k_idx in pool and k_idx not in used:
                        lo, hi = bounds.get(c, (0.0, 0.0))
                        v = rng.uniform(lo, hi) if hi > lo else 0.0
                        new_df.loc[k_idx, c] = v
                        used.add(k_idx)
                # remove used rows from the global pool
                if used:
                    pool = [i for i in pool if i not in used]
    
            outputs.append(new_df.reset_index(drop=True))
    
        return outputs

    def generate_local_perturbations_from_existing(
        time_col: str = "Time_seconds",
        n_per_original: int = 1,
        # magnitudes
        adjust_prob: float = 1.0,            # chance to adjust each existing non-zero bolus
        volume_scale: float = 0.15,          # multiplicative noise std: v <- v * (1 + N(0, volume_scale))
        time_jitter_s: float = 60.0,         # seconds of std-dev for time jitter when adding/swapping
        # adding a few extra small events
        add_prob: float = 0.3,               # per column probability to add events
        max_added_per_col: int = 1,
        add_magnitude_scale: float = 0.5,    # scale for added-event magnitude relative to typical for that column
        # swapping nearby events in time
        swap_prob: float = 0.3,              # per column probability to perform swaps
        max_swaps: int = 1,
        # constraints
        min_separation_s: float = 0.0,       # enforce min time between events in same column
        time_window: Tuple[Optional[float], Optional[float]] = (None, None),
        stretch: float = 0.5,                # for bounds computed from data
        manual_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        include_cols: Optional[List[str]] = None,   # if provided, only perturb these columns
        exclude_cols: Tuple[str, ...] = ("water",), # columns to skip (case-insensitive match)
        seed: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        """Generate local perturbations of each U in ``U_list``.
    
        Locality is controlled by small multiplicative volume noise, small time jitter,
        a tiny number of *additional* events near existing ones, and optional swaps of
        events to nearby time indices. Bounds per column are respected and can be
        overridden manually.
        """
        if not U_list:
            return []
    
        rng = random.Random(seed)
        # compute global bounds (then override)
        bounds = compute_per_column_bounds(U_list, time_col=time_col, stretch=stretch, manual_bounds=manual_bounds)
    
        outputs: List[pd.DataFrame] = []
    
        for df in U_list:
            if time_col not in df.columns:
                raise ValueError(f"Missing '{time_col}' column.")
            cols = [c for c in df.columns if c != time_col]
    
            # filter columns
            if include_cols is not None:
                cols = [c for c in cols if c in include_cols]
            if exclude_cols:
                ex_low = {c.lower() for c in exclude_cols}
                cols = [c for c in cols if c.lower() not in ex_low]
    
            # base time grid & step
            t = df[time_col].to_numpy(dtype=float)
            if t.size < 2:
                raise ValueError("Each U DataFrame must have at least 2 time points.")
            dt = float(np.median(np.diff(t)))
            K = t.size
    
            # allowed time window for new/shifted events
            win_lo = t[0] if time_window[0] is None else float(time_window[0])
            win_hi = t[-1] if time_window[1] is None else float(time_window[1])
            allowed_mask = (t >= win_lo) & (t <= win_hi)
            allowed_idx = np.where(allowed_mask)[0]
            if allowed_idx.size == 0:
                allowed_idx = np.arange(K)
    
            # step-based jitter & separation
            jitter_steps = max(1, int(round(time_jitter_s / max(dt, 1e-12))))
            sep_steps = max(1, int(round(min_separation_s / max(dt, 1e-12)))) if min_separation_s > 0 else 0
    
            for _ in range(n_per_original):
                pert = df.copy(deep=True)
    
                # 1) Adjust volumes of existing non-zero boluses
                for c in cols:
                    lo, hi = bounds.get(c, (0.0, np.inf))
                    col = pert[c].to_numpy(dtype=float)
                    nz = np.flatnonzero(col != 0.0)
                    for k in nz:
                        if rng.random() <= adjust_prob:
                            v = col[k]
                            v_new = v * (1.0 + rng.normalvariate(0.0, volume_scale))
                            v_new = max(0.0, v_new)
                            if math.isfinite(hi):
                                v_new = min(v_new, hi)
                            if math.isfinite(lo):
                                v_new = max(v_new, lo if v > 0 else 0.0)  # keep small but non-negative
                            col[k] = v_new
                    pert[c] = col
    
                # 2) Optionally add a few small extra events near existing ones
                for c in cols:
                    if max_added_per_col <= 0 or rng.random() > add_prob:
                        continue
                    lo, hi = bounds.get(c, (0.0, np.inf))
                    col = pert[c].to_numpy(dtype=float)
                    nz = np.flatnonzero(col != 0.0)
                    n_add = rng.randint(1, max_added_per_col)
    
                    # choose magnitude baseline
                    typical = float(np.median(col[nz])) if nz.size else (0.1 * hi if math.isfinite(hi) else 1.0)
                    typical = max(typical, 1e-9)
    
                    for _a in range(n_add):
                        # pick a base index near an existing event if possible
                        if nz.size:
                            base_k = int(rng.choice(nz))
                        else:
                            base_k = int(rng.choice(allowed_idx))
                        k_prop = int(np.clip(base_k + rng.randrange(-jitter_steps, jitter_steps + 1), 0, K - 1))
                        if k_prop not in allowed_idx:
                            continue
                        # magnitude: small, relative to typical
                        mult = max(0.0, 1.0 + rng.normalvariate(0.0, volume_scale))
                        v_new = typical * add_magnitude_scale * mult
                        # respect bounds
                        if math.isfinite(hi):
                            v_new = min(v_new, hi)
                        if v_new <= 0.0:
                            continue
                        col[k_prop] += v_new
    
                        # enforce min separation by merging if needed
                        if sep_steps > 0:
                            k_prev = max(0, k_prop - sep_steps)
                            k_next = min(K - 1, k_prop + sep_steps)
                            # collapse tiny clusters by summing into k_prop and zeroing neighbors
                            left = np.arange(k_prev, k_prop)
                            right = np.arange(k_prop + 1, k_next + 1)
                            for kk in np.concatenate([left, right]):
                                if col[kk] != 0.0:
                                    col[k_prop] += col[kk]
                                    col[kk] = 0.0
    
                    pert[c] = col
    
                # 3) Optionally swap one or two event times per column (local)
                for c in cols:
                    if max_swaps <= 0 or rng.random() > swap_prob:
                        continue
                    col = pert[c].to_numpy(dtype=float)
                    nz = np.flatnonzero(col != 0.0)
                    if nz.size == 0:
                        continue
                    n_swaps = rng.randint(1, max_swaps)
                    for _s in range(n_swaps):
                        k1 = int(rng.choice(nz))
                        k2_prop = int(np.clip(k1 + rng.randrange(-jitter_steps, jitter_steps + 1), 0, K - 1))
                        if k2_prop not in allowed_idx:
                            continue
                        # swap values at k1 and k2_prop
                        col[k1], col[k2_prop] = col[k2_prop], col[k1]
                    pert[c] = col
    
                outputs.append(pert.reset_index(drop=True))
    
        return outputs



    # ------------------------------------------------------------------
    # balance loss: mRNA & p* get weight, latent p gets 0
    # ------------------------------------------------------------------
    def compute_channel_weight(self, *, device="cuda"):
        channels = ("Broccoli [RFU]", "mCherry [RFU]")
        abs_vals = {ch: [] for ch in channels}
        for df in self.outputs_df.values():
            for ch in channels:
                abs_vals[ch].append(np.abs(df[ch].to_numpy()))
        mean_abs = torch.tensor([np.concatenate(abs_vals[ch]).mean()
                                 for ch in channels], dtype=torch.float32)
        base = mean_abs.max()
        w = torch.zeros(10)                       # (mRNA, p, p*)
        w[0] = 1.25
        w[1] = 0.          # p  ignored
        w[2] = 1
        w[3] = 0
        w[4] = 0
        w[5] = 0
        w[6] = 0
        w[7] = 0
        w[8] = 0
        w[9] = 0.1
        
        self.loss_weight = w.to(device)
    # ------------------------------------------------------------------
    # random train/test split
    # ------------------------------------------------------------------
    def make_train_test_split_simple(self, *, test_frac=0.2, seed=12):
        N = len(self.x0_list)
        rng = random.Random(seed)
        idx = list(range(N))
        rng.shuffle(idx)
        n_test = int(N * test_frac)
        self.test_idx, self.train_idx = idx[:n_test], idx[n_test:]
        print(f"Split made → {len(self.train_idx)} train | {len(self.test_idx)} test")
            
    def make_train_test_split(self, *, test_frac=0.125, val_frac=0.125):
        N = len(self.x0_list)
        self.choice = 57
        rng = random.Random(self.choice)
        idx = list(range(N))
        rng.shuffle(idx)
    
        n_test = int(N * test_frac)
        self.test_idx = idx[:n_test]
        rest = idx[n_test:]
    
        n_val = int(len(rest) * val_frac)
        self.val_idx = rest[:n_val] if n_val > 0 else []
        self.train_idx = rest[n_val:]
    
        msg = f"Split → {len(self.train_idx)} train"
        if self.val_idx: msg += f" | {len(self.val_idx)} val"
        msg += f" | {len(self.test_idx)} test"
        print(self.test_idx, self.val_idx, self.train_idx)
        

    def _make_subset_ds(self, idx_list):
        x0 = [self.x0_list[i]  for i in idx_list]
        u  = [self.u_list[i]   for i in idx_list]
        dna= [self.dna_list[i] for i in idx_list]
        dt = [self.dt_list[i]  for i in idx_list]
        y  = [self.y_list[i]   for i in idx_list]
        return Datastruct(x0, u, dna, dt, y)


    def load_ensemble_from_folder(
        self,
        folder: str | Path,
        n: int | None = None,
        pattern: str = "*.pt",
        sort_by: str = "name",          # "name" | "mtime"
        device: str | None = None,      # None → use self.device
        strict: bool = False,
        model_factory=None,             # e.g. lambda meta: GRU_model_latent_decay_simple_fb_mat(in_u=meta["in_u"], hidden=meta.get("hidden",128))
    ):
        """
        Load up to N models from `folder` into self.ensemble = {i: model_i}.
    
        Expects each checkpoint to be a dict with:
          - "state_dict": model parameters
          - "meta":       (optional) dict used by your factory to reconstruct the model
        """

        from pathlib import Path
        import torch
        from torch import nn
        import inspect

        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder}")
    
        files = list(folder.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching pattern '{pattern}' in {folder}")
    
        files.sort(key=(lambda p: p.stat().st_mtime) if sort_by == "mtime" else (lambda p: p.name))
        if n is not None:
            files = files[:int(n)]
    
        dev = torch.device(device) if device is not None else self.device
    
        # helper: build model from meta whether model_factory is a class or a callable
        def _build_from_meta(factory, meta: dict) -> nn.Module:
            if factory is None:
                raise ValueError("Provide model_factory (class or callable(meta)->nn.Module).")
    
            # If a class was passed (subclass of nn.Module), construct with kwargs from meta
            if isinstance(factory, type) and issubclass(factory, nn.Module):
                sig = inspect.signature(factory.__init__)
                kwargs = {}
    
                def add_param(name, value):
                    if name in sig.parameters:
                        kwargs[name] = value
    
                # Pull from meta with fallbacks
                in_u = int(meta.get("in_u", len(getattr(self, "input_cols", [])) - 1))
                if in_u < 0:
                    raise ValueError("Could not infer 'in_u'. Pass it in meta or set self.input_cols.")
                add_param("in_u", in_u)
                add_param("hidden", int(meta.get("hidden", 128)))
                add_param("num_layers", int(meta.get("num_layers", 2)))
                add_param("dropout", float(meta.get("dropout", 0.2)))
    
                return factory(**kwargs)
    
            # Otherwise assume it's a callable(meta) -> model
            return factory(meta)
    
        self.ensemble, self.ensemble_meta = {}, {}
        for i, ckpt_path in enumerate(files):
            ckpt = torch.load(ckpt_path, map_location=dev)
            if not (isinstance(ckpt, dict) and "state_dict" in ckpt):
                raise ValueError(f"Bad checkpoint format: {ckpt_path}")
            meta = ckpt.get("meta", {})
            model = _build_from_meta(model_factory, meta).to(dev)
            model.load_state_dict(ckpt["state_dict"], strict=strict)
            model.eval()
            self.ensemble[i] = model
            self.ensemble_meta[i] = {"path": str(ckpt_path), "meta": meta}
        return self.ensemble
    
    def MC_sample_U_enforce_coverage_with_first3(self,
        # U_list: List[pd.DataFrame],
        time_col: str = "Time_seconds",
        n_samples: int = 100,
        # where additions are allowed; None -> grid start/end
        time_window: Tuple[Optional[float], Optional[float]] = (None, None),
        # experiment duration: None -> largest final time across U_list
        duration_s: Optional[float] = None,
        # per-species cap (int for all, or dict per-col)
        max_events_per_col: Union[int, Dict[str, int]] = 3,
        # minimum separation (seconds) between boluses of the SAME species
        min_separation_s: float = 30.0,
        # global bounds via data then stretch by 50% (→ [min*(1-0.5), max*(1+0.5)])
        stretch: float = 0.5,
        # manual {col: (min, max)} overrides computed bounds
        manual_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        # probability to omit exactly one species entirely (others must appear ≥1 time)
        omit_one_prob: float = 0.5,
        # base grid choice: None -> first df’s (t0, dt); else provide (t0, dt)
        base_grid: Optional[Tuple[float, float]] = None,
        # ENFORCED FIRST-3:
        water_col: str = "water",
        fb_col: str = "FB",
        dna_col: str = "DNA",
        dna_fixed_volume: float = 240.0,
        seed: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        """
        Create `n_samples` new U tables with constraints:
          - First three additions (earliest allowed steps) are: Water → FB → DNA.
            DNA’s first addition is fixed to `dna_fixed_volume`.
          - At every time row, at most ONE species receives an addition.
          - Each species must have ≥1 event EXCEPT possibly one species, omitted
            with probability `omit_one_prob` (else all must appear ≥1 time).
          - Per-species max events enforced; per-species min separation enforced.
          - Bounds are computed globally once from U_list (then optionally overridden).
        """
        import math
    
        U_list = list(self.inputs_df.values())
        if not U_list:
            raise ValueError("U_list is empty.")
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
    
        # ---------- 1) Global columns & bounds ----------
        all_cols = sorted(set().union(*[df.columns for df in U_list]))
        if time_col not in all_cols:
            raise ValueError(f"Time column '{time_col}' not found in inputs.")
        data_cols = [c for c in all_cols if c != time_col]
    
        # case-insensitive map for requested enforced columns
        def _find_col(name: str) -> str:
            low = {c.lower(): c for c in data_cols}
            return low.get(name.lower(), name)  # fall back to given name
    
        water_col = _find_col(water_col)
        fb_col = _find_col(fb_col)
        dna_col = _find_col(dna_col)
    
        # ensure enforced columns are represented in data_cols
        for c in (water_col, fb_col, dna_col):
            if c not in data_cols:
                data_cols.append(c)
    
        bounds: Dict[str, Tuple[float, float]] = {}
        for c in data_cols:
            col_min = float(min(float(df[c].min()) if c in df.columns else 0.0 for df in U_list))
            col_max = float(max(float(df[c].max()) if c in df.columns else 0.0 for df in U_list))
            lo = max(0.0, col_min * (1.0 - stretch))
            hi = col_max * (1.0 + stretch)
            if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
                lo, hi = 0.0, 0.0
            bounds[c] = (lo, hi)
    
        if manual_bounds:
            for k, v in manual_bounds.items():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    bounds[k] = (float(v[0]), float(v[1]))
    
        # ---------- 2) Global duration & base grid ----------
        if duration_s is None:
            duration_s = max(float(df[time_col].iloc[-1]) for df in U_list)
    
        if base_grid is None:
            t_src = U_list[0][time_col].to_numpy(dtype=float)
            if t_src.size < 2:
                raise ValueError("Base grid df must have at least 2 time points.")
            dt = float(np.median(np.diff(t_src)))
            t0 = float(t_src[0])
        else:
            t0, dt = map(float, base_grid)
    
        n_steps = max(1, int(round((duration_s - t0) / max(dt, 1e-12))))
        t = t0 + np.arange(n_steps + 1) * dt
        K = t.size
    
        win_lo = t0 if time_window[0] is None else float(time_window[0])
        win_hi = (t0 + n_steps * dt) if time_window[1] is None else float(time_window[1])
        allowed_mask = (t >= win_lo) & (t <= win_hi)
        allowed_idx = np.where(allowed_mask)[0]
        allowed_idx = allowed_idx[allowed_idx < K - 1].astype(int)  # events for steps [0..K-2]
    
        if allowed_idx.size < 3:
            raise ValueError("Allowed time window doesn't include ≥3 steps to place Water→FB→DNA.")
    
        # per-col cap & min separation
        def _cap_for(col: str) -> int:
            if isinstance(max_events_per_col, int):
                return max(0, int(max_events_per_col))
            return max(0, int(max_events_per_col.get(col, 0)))
    
        step_gap = max(1, int(round(min_separation_s / max(dt, 1e-12))))
    
        def _pick_indices_for_species(pool: List[int], n_events: int) -> List[int]:
            if n_events <= 0 or not pool:
                return []
            chosen: List[int] = []
            avail = pool.copy()
            rng.shuffle(avail)
            while avail and len(chosen) < n_events:
                i = avail.pop()
                if all(abs(i - j) >= step_gap for j in chosen):
                    chosen.append(i)
                    avail = [u for u in avail if abs(u - i) >= step_gap]
            chosen.sort()
            return chosen
    
        # ---------- 3) Sampling loop ----------
        outputs: List[pd.DataFrame] = []
    
        for _ in range(n_samples):
            # Decide omission (0 or 1 species omitted)
            cols = data_cols.copy()
            rng.shuffle(cols) 
    
            if rng.random() < omit_one_prob and len(cols) >= 2:
                omit = rng.choice(cols)
                required_species = [c for c in cols if c != omit]
            else:
                omit = None
                required_species = cols[:]
    
            # Make sure the enforced three are *not* omitted.
            for c in (water_col, fb_col, dna_col):
                if omit == c:
                    # if we randomly chose to omit an enforced species, undo omission
                    omit = None
                    required_species = cols[:]
                    break
    
            taken_global: set[int] = set()
            events_per_species: Dict[str, List[int]] = {c: [] for c in data_cols}
    
            # ---- ENFORCE first three steps: earliest three allowed rows ----
            first3 = allowed_idx[:3].tolist()  # earliest chronological
            # 1) Water at first step
            events_per_species[water_col].append(first3[0])
            taken_global.add(first3[0])
            # 2) FB at second
            events_per_species[fb_col].append(first3[1])
            taken_global.add(first3[1])
            # 3) DNA at third (fixed volume later)
            events_per_species[dna_col].append(first3[2])
            taken_global.add(first3[2])
    
            # Ensure we don't exceed caps for those enforced species
            for enforced in (water_col, fb_col, dna_col):
                if _cap_for(enforced) == 0:
                    # override: force cap ≥1 to satisfy enforcement
                    pass  # we still keep the enforced event; later topping up will respect cap
    
            # ---- Guarantee ≥1 for each required species (besides those already covered) ----
            # Put enforced first to not double-count; then others.
            already_has_one = {water_col, fb_col, dna_col}
            for c in rng.sample([x for x in required_species if x not in already_has_one], k=len(required_species)-len(already_has_one)):
                free_rows = [k for k in allowed_idx if k not in taken_global]
                if not free_rows:
                    break
                if len(events_per_species[c]) == 0:
                    idx1 = _pick_indices_for_species(free_rows, 1)
                    if idx1:
                        events_per_species[c].extend(idx1)
                        taken_global.update(idx1)
    
            # ---- Top-up up to per-species caps (including enforced species) ----
            for c in rng.sample(cols, k=len(cols)):
                if omit is not None and c == omit:
                    continue
                cap = _cap_for(c)
                already = len(events_per_species[c])
                if cap <= already:
                    continue
                free_rows = [k for k in allowed_idx if k not in taken_global]
                if not free_rows:
                    break
                extra = rng.randint(0, cap - already)
                if extra <= 0:
                    continue
                idx_extra = _pick_indices_for_species(free_rows, extra)
                if idx_extra:
                    events_per_species[c].extend(idx_extra)
                    taken_global.update(idx_extra)
    
            # ---- Build DataFrame & assign magnitudes ----
            new_df = pd.DataFrame(0.0, index=np.arange(K), columns=[time_col] + data_cols)
            new_df[time_col] = t
    
            # Fill random magnitudes first, then override DNA first-step with fixed 240
            for c in data_cols:
                lo, hi = bounds.get(c, (0.0, 0.0))
                for k_idx in events_per_species[c]:
                    if c == dna_col and k_idx == first3[2]:
                        # Fixed DNA volume for the enforced step 3
                        new_df.at[k_idx, c] = float(dna_fixed_volume)
                    else:
                        if hi > lo:
                            new_df.at[k_idx, c] = rng.uniform(lo, hi)
                        else:
                            new_df.at[k_idx, c] = 0.0
    
            outputs.append(new_df.reset_index(drop=True))
    
        return outputs
        
    def MC_sample_windows_lysate_first(self,
        time_col: str = "Time_seconds",
        n_samples: int = 100,
        # timeline/grid
        duration_min: int = 60*24,               # 2 hours
        window_min: int = 15,                  # window spacing
        max_per_window: int = 5,               # up to 5 inputs per window
        within_window_spacing_min: int = 1,    # 1-minute spacing inside window
        # enforced species/volumes
        lysate_col: str = "Lysate 2%PEG",
        dna_col: str = "DNA",
        lysate_range_nl: Tuple[float, float] = (500.0, 5000.0),  # lysate only at start
        dna_fixed_nl: float = 240.0,                              # DNA one event, any time
        # per-event minima & global budget
        min_nl_all: float = 50.0,
        total_volume_cap_nl: float = 12000.0,
        # bounds (upper limits) handling
        stretch: float = 0.5,                  # from global data mins/maxs
        manual_bounds_nl: Optional[Dict[str, Tuple[float, float]]] = None,
        # base grid; if None, taken from first df (t0, dt seconds)
        base_grid: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        """
        Generate n_samples experiments with:
          - Lysate added only at the start (first available minute), volume ~ U[500,5000] nL,
          - DNA has one event with fixed 240 nL at any later slot,
          - Windows every 15 minutes; within each window up to 5 inputs, spaced 1 minute,
          - ≥1 event for every species, one event per time row globally,
          - Per-event minima 50 nL (lysate range overrides), total volume cap 12,000 nL,
          - All inputs within 2 hours (9 windows).
        """
        import math
        U_list = list(self.inputs_df.values())
        if not U_list:
            raise ValueError("U_list is empty.")
        rng = random.Random(seed)
    
        # ---------- 1) Global columns (species) ----------
        all_cols = sorted(set().union(*[df.columns for df in U_list]))
        if time_col not in all_cols:
            raise ValueError(f"Time column '{time_col}' not found.")
        species_cols = [c for c in all_cols if c != time_col]
    
        # case-insensitive mapping for required columns
        def _find_col(name: str) -> str:
            low = {c.lower(): c for c in species_cols}
            return low.get(name.lower(), name)
        lysate_col = _find_col(lysate_col)
        dna_col = _find_col(dna_col)
    
        # ensure columns exist in schema
        for c in (lysate_col, dna_col):
            if c not in species_cols:
                species_cols.append(c)
    
        # ---------- 2) Global bounds (from data; then manual override) ----------
        bounds: Dict[str, Tuple[float, float]] = {}
        for c in species_cols:
            col_min = float(min(float(df[c].min()) if c in df.columns else 0.0 for df in U_list))
            col_max = float(max(float(df[c].max()) if c in df.columns else 0.0 for df in U_list))
            lo = max(0.0, col_min * (1.0 - stretch))
            hi = col_max * (1.0 + stretch)
            if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
                lo, hi = 0.0, 0.0
            bounds[c] = (lo, hi)
    
        if manual_bounds_nl:
            for k, v in manual_bounds_nl.items():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    bounds[k] = (float(v[0]), float(v[1]))
    
        # helper to get per-event minimum (non-lysate)
        def _min_for(col: str) -> float:
            if col == dna_col:
                # DNA's enforced event is fixed 240; for any extras (if ever allowed),
                # keep at least min_nl_all
                return max(min_nl_all, bounds.get(col, (0.0, 0.0))[0])
            return max(min_nl_all, bounds.get(col, (0.0, 0.0))[0])
    
        # ---------- 3) Build a single global grid (from first df or base_grid) ----------
        if base_grid is None:
            t_src = U_list[0][time_col].to_numpy(dtype=float)
            if t_src.size < 2:
                raise ValueError("Base grid df must have at least 2 time points.")
            dt = float(np.median(np.diff(t_src)))      # seconds
            t0 = float(t_src[0])
        else:
            t0, dt = map(float, base_grid)
    
        # total steps (K rows)
        total_steps = int(round((duration_min * 60.0) / max(dt, 1e-12)))
        K = total_steps + 1
        t = t0 + np.arange(K) * dt
    
        # windows (start minutes): 0, 15, ..., 120
        window_starts_min = list(range(0, 150 + 1, window_min))
        # precompute slot indices per window (minute offsets 0..max_per_window-1)
        window_slots_idx = []
        for ws_min in window_starts_min:
            base_t = t0 + ws_min * 60.0
            base_idx = int(round((base_t - t0) / dt))
            step_stride = int(round((within_window_spacing_min * 60.0) / max(dt, 1e-12)))
            slots = [base_idx + i * step_stride for i in range(max_per_window)]
            slots = [s for s in slots if 0 <= s < K - 1]  # events placed on rows [0..K-2]
            window_slots_idx.append(slots)
    
        # ---------- 4) Sampling loop ----------
        outputs: List[pd.DataFrame] = []
    
        for _ in range(n_samples):
            # init blank frame
            new_df = pd.DataFrame(0.0, index=np.arange(K), columns=[time_col] + species_cols)
            new_df[time_col] = t
    
            # bookkeeping
            used_rows_global: set[int] = set()
            per_window_used: Dict[int, int] = {w: 0 for w in range(len(window_starts_min))}
            remaining_budget = float(total_volume_cap_nl)
    
            # helper to place one event
            def _place_event(col: str, row_idx: int, volume: float) -> bool:
                nonlocal remaining_budget
                if row_idx in used_rows_global:
                    return False
                if remaining_budget < volume:
                    return False
                hi = bounds.get(col, (0.0, float("inf")))[1]
                if hi > 0.0:
                    volume = min(volume, hi)
                new_df.at[row_idx, col] = float(volume)
                used_rows_global.add(row_idx)
                remaining_budget -= float(volume)
                # update window usage
                for wi, slots in enumerate(window_slots_idx):
                    if row_idx in slots:
                        per_window_used[wi] += 1
                        break
                return True
    
            # 4.a Enforce lysate at the very start (first window, first slot) — ONLY ONCE
            if not window_slots_idx or not window_slots_idx[0]:
                raise ValueError("No available slot in the first window to place lysate.")
            lysate_row = window_slots_idx[0][0]
            lys_lo, lys_hi = lysate_range_nl
            lys_lo = max(lys_lo, 0.0)
            lys_hi = max(lys_hi, lys_lo)
            lysate_vol = rng.uniform(lys_lo, lys_hi)
            if not _place_event(lysate_col, lysate_row, lysate_vol):
                # If budget too small or row taken (shouldn't be), return empty plan
                outputs.append(new_df.reset_index(drop=True))
                continue
    
            # 4.b Ensure one DNA event (fixed 240 nL), at any slot other than lysate_row
            dna_placed = False
            for wi, slots in enumerate(window_slots_idx):
                for s in slots:
                    if s == lysate_row:
                        continue
                    if random.random() < 0.05:
                        if _place_event(dna_col, s, float(dna_fixed_nl)):
                            dna_placed = True
                            break
                if dna_placed:
                    break
            if not dna_placed:
                outputs.append(new_df.reset_index(drop=True))
                continue
    
            # 4.c Ensure every species appears at least once (excluding ones already placed)
            placed_species = {lysate_col, dna_col}
            remaining_species = [c for c in species_cols if c not in placed_species]
            rng.shuffle(remaining_species)
    
            for col in remaining_species:
                placed = False
                # randomize window order to avoid bias
                for wi in rng.sample(range(len(window_slots_idx)), k=len(window_slots_idx)):
                    if per_window_used[wi] >= max_per_window:
                        continue
                    for s in window_slots_idx[wi]:
                        if s in used_rows_global:
                            continue
                        v_min = _min_for(col)
                        lo, hi = bounds.get(col, (0.0, 0.0))
                        v_hi = hi if hi > v_min else max(v_min, hi)
                        vol = v_min if v_hi <= v_min else rng.uniform(v_min, v_hi)
                        if _place_event(col, s, vol):
                            placed = True
                            break
                    if placed:
                        break
                # if not placed, budget/window limits prevented it; we proceed anyway
    
            # 4.d Optionally add more random events while budget and slots remain
            all_slots = [(wi, s)
                         for wi, slots in enumerate(window_slots_idx)
                         for s in slots]
            rng.shuffle(all_slots)
    
            for wi, s in all_slots:
                if remaining_budget <= min_nl_all:
                    break
                if per_window_used[wi] >= max_per_window:
                    continue
                if s in used_rows_global:
                    continue
                # pick a random species, but NEVER lysate again
                c = rng.choice([sp for sp in species_cols if sp != lysate_col])
                v_min = _min_for(c)
                lo, hi = bounds.get(c, (0.0, 0.0))
                v_hi = hi if hi > v_min else max(v_min, hi)
                vol = v_min if v_hi <= v_min else rng.uniform(v_min, v_hi)
                _place_event(c, s, vol)
    
            outputs.append(new_df.reset_index(drop=True))
    
        return outputs

 
    def MC_sample_U_enforce_coverage(self,
        time_col: str = "Time_seconds",
        n_samples: int = 100,
        # where additions are allowed; None means use grid start/end
        time_window: Tuple[Optional[float], Optional[float]] = (None, None),
        # experiment duration: None -> use largest final time across U_list
        duration_s: Optional[float] = None,
        # per-species cap (int for all, or dict per-col)
        max_events_per_col: Union[int, Dict[str, int]] = 3,
        # minimum separation (seconds) between boluses of the SAME species
        min_separation_s: float = 30.0,
        # compute bounds from data then stretch by 50% (i.e., [min*(1-0.5), max*(1+0.5)])
        stretch: float = 0.5,
        # manual {col: (min, max)} overrides computed bounds
        manual_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        # probability to omit exactly one species entirely (others must appear ≥1 time)
        omit_one_prob: float = 0.5,
        # base grid choice: None -> use the first df's dt/t0; else provide (t0, dt)
        base_grid: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        """
        Create `n_samples` new U tables by randomly sampling bolus additions per column,
        with constraints:
          - At every time row, at most ONE species receives an addition.
          - Each species must have ≥1 event EXCEPT possibly one species, which is
            omitted with probability `omit_one_prob` (else all must appear).
          - Per-species max events enforced; per-species min separation enforced.
    
        Notes
        -----
        * Bounds are computed GLOBALLY across all U_list then (optionally) overridden.
        * Sampling uses a SINGLE time grid (not per original df):
            t0, dt from `base_grid` or from the FIRST df in U_list;
            duration from `duration_s` or maximum final time across U_list.
        """
        import math
    
        U_list = list(self.inputs_df.values())
        if not U_list:
            raise ValueError("U_list is empty.")
    
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
    
        # ---------- 1) Global columns and bounds ----------
        # union of all columns present in inputs
        all_cols = sorted(set().union(*[df.columns for df in U_list]))
        if time_col not in all_cols:
            raise ValueError(f"Time column '{time_col}' not found in inputs.")
        data_cols = [c for c in all_cols if c != time_col]
    
        # global min/max then stretch
        bounds: Dict[str, Tuple[float, float]] = {}
        for c in data_cols:
            col_min = float(min(float(df[c].min()) if c in df.columns else 0.0 for df in U_list))
            col_max = float(max(float(df[c].max()) if c in df.columns else 0.0 for df in U_list))
            lo = max(0.0, col_min * (1.0 - stretch))
            hi = col_max * (1.0 + stretch)
            if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
                lo, hi = 0.0, 0.0
            bounds[c] = (lo, hi)
    
        if manual_bounds:
            for k, v in manual_bounds.items():
                if k in bounds and isinstance(v, (list, tuple)) and len(v) == 2:
                    bounds[k] = (float(v[0]), float(v[1]))
    
        # ---------- 2) Global duration and base grid ----------
        if duration_s is None:
            duration_s = max(float(df[time_col].iloc[-1]) for df in U_list)
    
        if base_grid is None:
            # derive (t0, dt) from the FIRST df
            t_src = U_list[0][time_col].to_numpy(dtype=float)
            if t_src.size < 2:
                raise ValueError("Base grid df must have at least 2 time points.")
            dt = float(np.median(np.diff(t_src)))
            t0 = float(t_src[0])
        else:
            t0, dt = map(float, base_grid)
    
        # build time grid up to global duration
        n_steps = max(1, int(round((duration_s - t0) / max(dt, 1e-12))))
        t = t0 + np.arange(n_steps + 1) * dt
        K = t.size
    
        # allowed rows within [win_lo, win_hi]
        win_lo = t0 if time_window[0] is None else float(time_window[0])
        win_hi = (t0 + n_steps * dt) if time_window[1] is None else float(time_window[1])
        allowed_mask = (t >= win_lo) & (t <= win_hi)
        allowed_idx = np.where(allowed_mask)[0]  # indices into [0..K-1]
    
        # time rows that accept events: usually we use the Δ rows ([:-1]); still enforce on [0..K-1)
        # We'll target indices in [0..K-1) since events apply to the step that starts at row k.
        allowed_idx = allowed_idx[allowed_idx < K - 1]
        allowed_idx = allowed_idx.astype(int)
    
        # unify per-col max events
        def _cap_for(col: str) -> int:
            if isinstance(max_events_per_col, int):
                return max(0, int(max_events_per_col))
            return max(0, int(max_events_per_col.get(col, 0)))
    
        # per-species min-gap measured in steps
        step_gap = max(1, int(round(min_separation_s / max(dt, 1e-12))))
    
        # helper: choose event indices for a single species given remaining free rows
        def _pick_indices_for_species(
            pool: List[int], n_events: int
        ) -> List[int]:
            if n_events <= 0 or not pool:
                return []
            # Greedy: pick random centers respecting step_gap within this species
            chosen: List[int] = []
            avail = pool.copy()
            rng.shuffle(avail)
            while avail and len(chosen) < n_events:
                i = avail.pop()
                # enforce separation *within this species*
                if all(abs(i - j) >= step_gap for j in chosen):
                    chosen.append(i)
                    # also prune too-close indices from the candidate list
                    avail = [u for u in avail if abs(u - i) >= step_gap]
            chosen.sort()
            return chosen
    
        # ---------- 3) Sampling loop (no loop over U_list here) ----------
        outputs: List[pd.DataFrame] = []
    
        for _ in range(n_samples):
            # Decide which species must appear:
            cols = data_cols.copy()
            rng.shuffle(cols)
    
            if rng.random() < omit_one_prob and len(cols) >= 2:
                # choose exactly one to omit entirely
                omit = rng.choice(cols)
                required_species = [c for c in cols if c != omit]
            else:
                omit = None
                required_species = cols[:]  # all must appear ≥1
    
            # we will try to assign ≥1 event to each species in `required_species`
            # respecting: one event per time row globally
            taken_global: set[int] = set()
            events_per_species: Dict[str, List[int]] = {c: [] for c in data_cols}
    
            # First pass: guarantee ≥1 for each required species, if possible
            # Work on a shuffled order to reduce bias
            for c in rng.sample(required_species, k=len(required_species)):
                # free rows are allowed rows not yet used globally
                free_rows = [k for k in allowed_idx if k not in taken_global]
                if not free_rows:
                    # not enough room to satisfy everyone; skip (violates strictness but keeps feasibility)
                    continue
                idx1 = _pick_indices_for_species(free_rows, 1)
                if idx1:
                    events_per_species[c].extend(idx1)
                    taken_global.update(idx1)
    
            # Second pass: top-up with extra events up to cap, still one per row globally
            for c in rng.sample(cols, k=len(cols)):
                if omit is not None and c == omit:
                    continue
                cap = _cap_for(c)
                already = len(events_per_species[c])
                # we want between already (≥1 if required) and cap
                if cap <= already:
                    continue
                # remaining rows available
                free_rows = [k for k in allowed_idx if k not in taken_global]
                if not free_rows:
                    continue
                # choose how many extra (could be 0)
                extra = rng.randint(0, cap - already)
                if extra <= 0:
                    continue
                idx_extra = _pick_indices_for_species(free_rows, extra)
                if idx_extra:
                    events_per_species[c].extend(idx_extra)
                    taken_global.update(idx_extra)
    
            # Build the DataFrame
            new_df = pd.DataFrame(0.0, index=np.arange(K), columns=[time_col] + data_cols)
            new_df[time_col] = t
    
            # sample magnitudes
            for c in data_cols:
                lo, hi = bounds.get(c, (0.0, 0.0))
                if hi <= lo:
                    continue
                for k_idx in events_per_species[c]:
                    val = rng.uniform(lo, hi)
                    new_df.at[k_idx, c] = val
    
            outputs.append(new_df.reset_index(drop=True))
        return outputs
    
    def MC_sample_windows_many_small_additions(
        self,
        time_col: str = "Time_seconds",
        n_samples: int = 100,
        # full experiment duration (24 h grid)
        duration_min: int = 24 * 60,
        # dosing allowed only in first 2 h
        dosing_window_min: int = 120,
        window_min: int = 5,                 # window spacing (min)
        max_per_window: int = 2,              # up to 5 inputs per window
        within_window_spacing_min: int = 1,   # spacing inside window (min)
        # enforced species/volumes
        lysate_col: str = "Lysate 2%PEG",
        water_col:  str = 'water',
        dna_col: str = "DNA",
        lysate_range_nl: Tuple[float, float] = (1120.,4250.0),  # lysate only once (first slot)
        water_range_nl: Tuple[float, float] = (750.,4250.0),  # lysate only once (first slot)
        dna_fixed_nl: float = 100.0,                              # DNA only once, right after lysate
        # per-event minima & global budget
        min_nl_all: float = 50.0,            # minimum per event for all species
        total_volume_cap_nl: float = 12000.0,
        # per-event small-additions upper cap for non-DNA/non-lysate species
        max_event_nl: float = 750.0,         # tune to make additions “small”
        # bounds handling
        stretch: float = 0.5,                # widen data-driven bounds
        manual_bounds_nl: Optional[Dict[str, Tuple[float, float]]] = None,
        # base grid; if None, taken from first df (t0, dt seconds)
        base_grid: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        """
        Generate n_samples experiments with:
          - 24 h time grid, but all additions are within first 2 h (dosing_window_min),
          - Lysate added at very first available slot only once (U[500,5000] nL),
          - DNA added exactly once in the NEXT available slot after lysate (240 nL),
          - One event per time row globally,
          - Every remaining species is added many times in small volumes (min 50 nL, max max_event_nl),
          - Stop when total volume hits the cap (default 12,000 nL),
          - Windows every 15 min; within a window up to 5 inputs, spaced 1 min.
        """
        import math, random
        import numpy as np
        import pandas as pd
        from typing import Dict, List, Optional, Tuple
    
        U_list = list(self.inputs_df.values())
        if not U_list:
            raise ValueError("U_list is empty.")
        rng = random.Random(seed)
    
        # ---------- 1) Species columns ----------
        all_cols = sorted(set().union(*[df.columns for df in U_list]))
        if time_col not in all_cols:
            raise ValueError(f"Time column '{time_col}' not found.")
        species_cols = [c for c in all_cols if c != time_col]
    
        # case-insensitive name resolution
        def _find_col(name: str) -> str:
            low = {c.lower(): c for c in species_cols}
            return low.get(name.lower(), name)
    
        lysate_col = _find_col(lysate_col)
        dna_col = _find_col(dna_col)
    
        # ensure required columns exist
        for c in (lysate_col, dna_col):
            if c not in species_cols:
                species_cols.append(c)
    
        # ---------- 2) Bounds ----------
        bounds: Dict[str, Tuple[float, float]] = {}
        for c in species_cols:
            col_min = float(min(float(df[c].min()) if c in df.columns else 0.0 for df in U_list))
            col_max = float(max(float(df[c].max()) if c in df.columns else 0.0 for df in U_list))
            lo = max(0.0, col_min * (1.0 - stretch))
            hi = col_max * (1.0 + stretch)
            if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
                lo, hi = 0.0, 0.0
            bounds[c] = (lo, hi)
    
        if manual_bounds_nl:
            for k, v in manual_bounds_nl.items():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    bounds[k] = (float(v[0]), float(v[1]))
    
        def _cap_volume(col: str, v: float) -> float:
            lo, hi = bounds.get(col, (0.0, float("inf")))
            if hi > 0.0:
                v = min(v, hi)
            return max(0.0, v)
    
        # ---------- 3) Global 24 h grid ----------
        if base_grid is None:
            t_src = U_list[0][time_col].to_numpy(dtype=float)
            if t_src.size < 2:
                raise ValueError("Base grid df must have at least 2 time points.")
            dt = float(np.median(np.diff(t_src)))  # seconds
            t0 = float(t_src[0])
        else:
            t0, dt = map(float, base_grid)
    
        total_steps = int(round((duration_min * 60.0) / max(dt, 1e-12)))
        K = total_steps + 1
        t = t0 + np.arange(K) * dt
    
        # ---------- 4) Build dosing windows (first 2 h only) ----------
        dosing_window_min = max(0, int(dosing_window_min))
        dosing_window_min = min(dosing_window_min, duration_min)
        window_starts_min = list(range(0, dosing_window_min + 1, window_min))
        window_starts_min = [i for i in window_starts_min if i not in [15,20,25,45,50,55,75,80,85,105,110,115]]
    
        window_slots_idx: List[List[int]] = []
        for ws_min in window_starts_min:
            base_t = t0 + ws_min * 60.0
            base_idx = int(round((base_t - t0) / dt))
            step_stride = int(round((within_window_spacing_min * 60.0) / max(dt, 1e-12)))
            slots = [base_idx + i * step_stride for i in range(max_per_window)]
            slots = [s for s in slots if 0 <= s < K - 1]  # events on rows [0..K-2]
            window_slots_idx.append(slots)
    
        # helper: list of all dosing slots
        all_slots = [(wi, s) for wi, slots in enumerate(window_slots_idx) for s in slots]
    
        outputs: List[pd.DataFrame] = []
    
        for _ in range(n_samples):
            # blank plan
            new_df = pd.DataFrame(0.0, index=np.arange(K), columns=[time_col] + species_cols)
            new_df[time_col] = t
    
            used_rows: set[int] = set()
            per_window_used = {wi: 0 for wi in range(len(window_slots_idx))}
            remaining_budget = float(total_volume_cap_nl)
    
            # place helper
            def _place(col: str, row_idx: int, vol: float) -> bool:
                nonlocal remaining_budget
                if row_idx in used_rows:
                    return False
                vol = _cap_volume(col, vol)
                if vol < min_nl_all:
                    vol = min_nl_all
                if remaining_budget < vol:
                    return False
                new_df.at[row_idx, col] = float(vol)
                used_rows.add(row_idx)
                remaining_budget -= float(vol)
                for wi, slots in enumerate(window_slots_idx):
                    if row_idx in slots:
                        per_window_used[wi] += 1
                        break
                return True
    
            # --- Lysate at very first slot of first window ---
            if not window_slots_idx or not window_slots_idx[0]:
                raise ValueError("No slot available in the first window.")
            lys_slot = window_slots_idx[0][0]
            lys_lo, lys_hi = lysate_range_nl
            lys_lo, lys_hi = max(lys_lo, 0.0), max(lys_hi, lys_lo)
            lys_vol = rng.uniform(lys_lo, lys_hi)
            if not _place(lysate_col, lys_slot, lys_vol):
                outputs.append(new_df.reset_index(drop=True))
                continue
    
            
            # --- Lysate at very first slot of first window ---
            if not window_slots_idx or not window_slots_idx[0]:
                raise ValueError("No slot available in the first window.")
            w_slot = window_slots_idx[0][1]
            wlo, whi = water_range_nl
            wl, wh = max(wlo, 0.0), max(whi, lys_lo)
            w_vol = rng.uniform(wl, wh)
            if not _place(water_col, w_slot, w_vol):
                outputs.append(new_df.reset_index(drop=True))
                continue
            
            # --- DNA exactly once, in the next available dosing slot ---
            dna_placed = False
            # Prefer the next slot in the first window if free; otherwise scan forward
            candidate_slots = []
            # next slot in first window (after lys_slot), then all remaining slots
            candidate_slots.extend([s for s in window_slots_idx[0] if s != lys_slot])
            for wi in range(1, len(window_slots_idx)):
                candidate_slots.extend(window_slots_idx[wi])
    
            for s in candidate_slots:
                if s in used_rows:
                    continue
                if _place(dna_col, s, float(dna_fixed_nl)):
                    dna_placed = True
                    break
            if not dna_placed:
                outputs.append(new_df.reset_index(drop=True))
                continue
    
            # --- Ensure each remaining species has at least one event (small volume) ---
            remaining_species = [c for c in species_cols if c not in (lysate_col, dna_col)]
            rng.shuffle(remaining_species)
    
            # build a shuffled list of free dosing slots
            free_slots = [s for (_, s) in all_slots if s not in used_rows]
            rng.shuffle(free_slots)
    
            def _draw_small(col: str) -> float:
                # small addition between [min_nl_all, max_event_nl], capped by bounds
                v = rng.uniform(min_nl_all, max_event_nl)
                return _cap_volume(col, v)
    
            for col in remaining_species:
                placed = False
                for s in free_slots:
                    if s in used_rows:
                        continue
                    if _place(col, s, _draw_small(col)):
                        placed = True
                        break
                # if not placed due to budget/slots, we move on
    
            # --- Sprinkle MANY small additions for all non-DNA species until budget exhausted ---
            # never add lysate or DNA again
            sprinkle_species = [c for c in species_cols if c not in (lysate_col, dna_col)]
            # create a long randomized iteration over remaining dosing slots
            free_slots = [s for (_, s) in all_slots if s not in used_rows]
            rng.shuffle(free_slots)
    
            for s in free_slots:
                if remaining_budget <= min_nl_all:
                    break
                # pick a random species (non-DNA / non-lysate)
                c = rng.choice(sprinkle_species)
                v = _draw_small(c)
                _place(c, s, v)
    
            outputs.append(new_df.reset_index(drop=True))
    
        return outputs
          
    def simulate_ensemble2(
        self,
        *,
        n_per_original: int = 25000,
        time_window: tuple[float | None, float | None] = (0.0, 60.0),
        time_col: str = "Time_seconds",
        dna_candidates: tuple[str, ...] = ("DNA c", "DNA", "dna", "DNA_c"),
        teacher_forcing: bool = False,
        return_params: bool = True,
        to_numpy: bool = True,
    ):
        """
        Monte-Carlo sample U, then simulate each U with every model in self.ensemble.
        No gradients. Formats inputs like the original solver.
        Saves results in self.mc_predict = {u_idx: {model_id: (pred, params_or_None)}}.
        """
        import numpy as np
        import torch
        import pickle
        import os
        # assert hasattr(self, "ensemble") and self.ensemble, "Load models into self.ensemble first."
    
        # 1) Make samples once
        self.sample = self.MC_sample_windows_many_small_additions(n_samples=n_per_original)
    
        dev = self.device
        # Ensure models are on device + eval
        for m in self.ensemble.values():
            m.to(dev).eval()
    
        self.mc_predict: dict[int, dict[int, tuple]] = {}
    
        for u_idx, df in enumerate(self.sample):
            print(u_idx)
            # ---- sort by time, sanity checks
            
            if time_col not in df.columns:
                raise ValueError(f"Sample {u_idx} is missing '{time_col}'.")
            df = df.sort_values(time_col).reset_index(drop=True)
    
            t = df[time_col].to_numpy(dtype=float)
            if t.size < 2:
                # skip too-short sequences
                continue
    
            # ---- find DNA column
            dna_col = next((c for c in dna_candidates if c in df.columns), None)
            if dna_col is None:
                raise ValueError(f"Sample {u_idx} has no DNA column; tried {dna_candidates}.")
    
            # ---- build column order: sorted controls (excl. time & DNA), then DNA
            cols_sorted = sorted([c for c in df.columns if c != time_col])
            if dna_col not in cols_sorted:
                cols_sorted.append(dna_col)  # safety
            ctrl_cols = [c for c in cols_sorted if c != dna_col]
    
            # ---- per-step arrays (rows 0..K-1), like your training pipeline
            K = t.size - 1
            dt_np = (t[1:] - t[:-1]).astype(np.float32)  # (K,)
            diffs_all = df[ctrl_cols + [dna_col]].to_numpy(dtype=float)  # (K+1, U_ctrl + 1)
            diffs_all = diffs_all[:-1, :]                                # (K,    U_ctrl + 1)
    
            u_ctrl = diffs_all[:, :-1].astype(np.float32)  # (K, U_ctrl)
            dna_raw = diffs_all[:, -1:].astype(np.float32) # (K, 1)
            dna_raw = dna_raw*0
            dna_raw[1] = 2.5
            
            # ---- optional scaling of controls (DNA excluded)
            # u_ctrl = self.scaler.transform(u_ctrl)
    
            # ---- tensors (B=1)
            x0      = torch.zeros((1, 3), dtype=torch.float32, device=dev)   # [mm0, p0, pm0] all zeros
            u_seq   = torch.tensor(u_ctrl, dtype=torch.float32, device=dev).unsqueeze(0)   # (1,K,U)
            dna_seq = torch.tensor(dna_raw, dtype=torch.float32, device=dev).unsqueeze(0)  # (1,K,1)
            dt_seq  = torch.tensor(dt_np,  dtype=torch.float32, device=dev).unsqueeze(0)   # (1,K)
            y_dummy = torch.zeros((1, K, 3), dtype=torch.float32, device=dev)              # (1,K,3)
    
            self.mc_predict = {'Inputs':df}

            
            with torch.no_grad():
                for mid, model in self.ensemble.items():
                    out = model(x0, u_seq, dna_seq, dt_seq, y_dummy, teacher_forcing=False)
                    if isinstance(out, tuple) and len(out) == 2:
                        pred, params = out
                    else:
                        pred, params = out, None
    
                    pred_out = pred.squeeze(0).detach().cpu()
                    params_out = None
                    if return_params:
                        params_out = None if params is None else params.squeeze(0).detach().cpu()
   
                    pred_out = pred_out.numpy()
                    if params_out is not None:
                        params_out = params_out.numpy()
                    
                    self.mc_predict[mid] = (pred_out, params_out)
    
    
            folder = Path('/home/bob-van-sluijs/Desktop/Exp 9 MC/')
            folder.mkdir(parents=True, exist_ok=True)
        
            # Use plain listdir count as requested
            N = len(os.listdir(folder))
            fname = f"exp_{N}.pkl"
            fpath = folder / fname
        
            with open(fpath, "wb") as f:
                pickle.dump(self.mc_predict, f, protocol=pickle.HIGHEST_PROTOCOL)

        return self.mc_predict
    
        
    def train_val(self, *, epochs=200, lr=1e-3,
                     hidden=128, num_layers=1,
                     model_cls=None, batch=150, normalize=False,decay = 0.0005,
                     save_path: str | Path = None, teacher_forcing = True,
                     val_frac: float = 0.1):   # ← NEW: hold-out from training as validation
        assert hasattr(self, "train_idx"), "Call make_train_test_split() first."
    
        # --- split current train_idx into train/val (shuffle already done earlier) ---
        if val_frac > 0.0 and len(self.train_idx) > 1:
            n_val = max(1, int(len(self.train_idx) * val_frac))
            val_idx   = self.train_idx[:n_val]
            train_idx = self.train_idx[n_val:]
        else:
            val_idx, train_idx = [], list(self.train_idx)
    
        train_ds = self._make_subset_ds(train_idx)
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                                  num_workers=0, collate_fn=collate,
                                  pin_memory=True)
    
        val_loader = None
        if len(val_idx) > 0:
            val_ds = self._make_subset_ds(val_idx)
            val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                                    num_workers=0, collate_fn=collate,
                                    pin_memory=True)
    
        model = model_cls(len(self.input_cols) - 1,
                          hidden = hidden, num_layers = num_layers).to(self.device)
        
        try:
            model = torch.jit.script(model)
            print('The model compiled successfully')
        except AttributeError:
            print('The model did not compile please check')
    
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
        best_val = float('inf')
        self.best_model = copy.deepcopy(model)
        for ep in range(1, epochs + 1):
            if ep == 100:
                teacher_forcing = False
            # ------------------------- TRAIN -------------------------
            model.train()
            train_total = 0.0
            for x0, u_seq, dna_seq, dt_seq, y_seq, lengths in train_loader:
                x0, u_seq, dna_seq, dt_seq, y_seq, lengths = (
                    x0.to(self.device), u_seq.to(self.device),
                    dna_seq.to(self.device), dt_seq.to(self.device),
                    y_seq.to(self.device), lengths.to(self.device)
                )
    
                opt.zero_grad()
                pred, params = model(x0, u_seq, dna_seq, dt_seq, y_seq, teacher_forcing = teacher_forcing)
    
                # ---- loss (unchanged) ----
                y_clamped    = y_seq.clamp_min(1.0)
                pred_clamped = pred.clamp_min(1.0)
                log_y, log_pred = torch.log1p(y_clamped), torch.log1p(pred_clamped)
    
                # NOTE: params layout is [VTX, kdm, VTL, kmt, kmtm, R, lambda]
                mean_st = y_seq.mean(dim=-1, keepdim=True)
                VTX_raw    = params[..., 0:1] / 0.125
                kdm_raw    = params[..., 1:2] / 0.01
                VTL_raw    = params[..., 2:3] / 0.075
                kmt_raw    = params[..., 3:4] / 0.00035
                kmtm_raw   = params[..., 4:5] / 0.0035
                R_raw      = params[..., 5:6] / 1

                VTX_state  = (VTX_raw * R_raw * mean_st).clamp_min(0.0)
                kdm_state  = (kdm_raw * mean_st).clamp_min(0.0)
                VTL_state  = (VTL_raw * R_raw * mean_st).clamp_min(0.0)
                kmt_state  = (kmt_raw * mean_st).clamp_min(0.0)
                kmtm_state = (kmtm_raw * mean_st).clamp_min(0.0)
                R_state    = (R_raw * mean_st).clamp_min(0.0)
                
                log_VTX     = torch.log1p(VTX_state)
                log_kdm     = torch.log1p(kdm_state)
                log_VTL     = torch.log1p(VTL_state)
                log_kmt     = torch.log1p(kmt_state)
                log_kmtkm   = torch.log1p(kmtm_state)
                log_R       = torch.log1p(R_state)

                log_pred_all = torch.cat([log_pred, log_VTX, log_kdm, log_VTL, log_kmt, log_kmtkm,  log_R], dim=-1)
                log_zero     = torch.zeros_like(log_VTX)
                log_y_all    = torch.cat([log_y, log_zero, log_zero,log_zero,log_zero,log_zero,log_zero], dim=-1)
    
                B, K, _ = y_seq.shape
                t_grid  = torch.arange(K, device=y_seq.device)
                valid_mask = (t_grid[None, :] < lengths[:, None]).unsqueeze(-1)
                cut_idx    = (lengths.float() * 0.95).long()
                tail_mask  = (t_grid[None, :] >= cut_idx[:, None]) & valid_mask.squeeze(-1)
                
                # time weighting ONLY for channel 2 (second half gets weight N)
                N = 3
                half_idx = lengths // 2
                is_second_half = (t_grid[None, :] >= half_idx[:, None]).unsqueeze(-1)     # (B,K,1) bool
                w_time = torch.where(
                    is_second_half,
                    torch.full_like(valid_mask, N, dtype=log_y_all.dtype),                 # (B,K,1) float
                    torch.ones_like(valid_mask, dtype=log_y_all.dtype)
                ) * valid_mask.to(torch.float32)      
                
                w = torch.zeros_like(log_y_all, dtype=log_y.dtype)
                w[..., 0] = valid_mask.squeeze(-1)  # mRNA
                w[..., 1] = tail_mask               # p
                w[..., 2] = (w_time * valid_mask).squeeze(-1)  # p*
                w[..., 3] = tail_mask               # VTX
                w[..., 4] = tail_mask               # kdm
                w[..., 5] = tail_mask               # VTL
                w[..., 6] = tail_mask               # kmt
                w[..., 7] = tail_mask               # kmtm
                w[..., 8] = tail_mask               # R proxy
        
                last_indices = (lengths - 1).clamp_min(0)
                batch_idx    = torch.arange(B, device=y_seq.device)
                y_last       = log_y_all[batch_idx, last_indices, 2]
                pred_last    = log_pred_all[batch_idx, last_indices, 2]
                extra_mse_chan    = ((pred_last - y_last) ** 2).mean(0, keepdim=True)
                err2     = (log_pred_all - log_y_all).pow(2) * w
                mse_chan = err2.sum((0, 1)) / w.sum((0, 1)).clamp_min(1)    
                mse_chan = torch.cat([mse_chan,extra_mse_chan])                        
                
                weighted = self.loss_weight * mse_chan
                mask = self.loss_weight.ne(0)# boolean mask of active terms
                den  = mask.sum().clamp_min(1).to(weighted.dtype)  # avoid /0
                loss = weighted[mask].sum() / den
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
                train_total += float(loss.item())
    
            # ---------------------- VALIDATION -----------------------
            if val_loader is not None:
                model.eval()
                val_total = 0.0
                with torch.no_grad():
                    for x0, u_seq, dna_seq, dt_seq, y_seq, lengths in val_loader:
                        x0, u_seq, dna_seq, dt_seq, y_seq, lengths = (
                            x0.to(self.device), u_seq.to(self.device),
                            dna_seq.to(self.device), dt_seq.to(self.device),
                            y_seq.to(self.device), lengths.to(self.device)
                        )
    
                        pred, params = model(x0, u_seq, dna_seq, dt_seq, y_seq, teacher_forcing = False)
                        y_clamped    = y_seq.clamp_min(1.0)
                        pred_clamped = pred.clamp_min(1.0)
                        log_y, log_pred = torch.log1p(y_clamped), torch.log1p(pred_clamped)
    
                        mean_st = y_seq.mean(dim=-1, keepdim=True)
                        VTX_raw    = params[..., 0:1] / 0.1
                        kdm_raw    = params[..., 1:2] / 0.01
                        VTL_raw    = params[..., 2:3] / 0.075
                        kmt_raw    = params[..., 3:4] / 0.00035
                        kmtm_raw   = params[..., 4:5] / 0.0035
                        R_raw      = params[..., 5:6] / 1
        
                        VTX_state  = VTX_raw * R_raw * mean_st
                        kdm_state  = kdm_raw * mean_st
                        VTL_state  = VTL_raw * R_raw * mean_st
                        kmt_state  = kmt_raw * mean_st
                        kmtm_state = kmtm_raw * mean_st
                        R_state    = R_raw * mean_st
                        
                        log_VTX     = torch.log1p(VTX_state)
                        log_kdm     = torch.log1p(kdm_state)
                        log_VTL     = torch.log1p(VTL_state)
                        log_kmt     = torch.log1p(kmt_state)
                        log_kmtkm   = torch.log1p(kmtm_state)
                        log_R       = torch.log1p(R_state)
        
                        log_pred_all = torch.cat([log_pred, log_VTX, log_kdm, log_VTL, log_kmt, log_kmtkm,  log_R], dim=-1)
                        log_zero     = torch.zeros_like(log_VTX)
                        log_y_all    = torch.cat([log_y, log_zero, log_zero,log_zero,log_zero,log_zero,log_zero], dim=-1)
            
                        B, K, _ = y_seq.shape
                        t_grid  = torch.arange(K, device=y_seq.device)
                        valid_mask = (t_grid[None, :] < lengths[:, None]).unsqueeze(-1)
                        cut_idx    = (lengths.float() * 0.95).long()
                        tail_mask  = (t_grid[None, :] >= cut_idx[:, None]) & valid_mask.squeeze(-1)
            
                        last_indices = (lengths - 1).clamp_min(0)
                        batch_idx    = torch.arange(B, device=y_seq.device)
                        y_last       = log_y_all[batch_idx, last_indices, 2]
                        pred_last    = log_pred_all[batch_idx, last_indices, 2]
                        extra_mse_chan = ((pred_last - y_last) ** 2).mean(0, keepdim=True)
                        
                        N =3
                        half_idx = lengths // 2
                        is_second_half = (t_grid[None, :] >= half_idx[:, None]).unsqueeze(-1)     # (B,K,1) bool
                        w_time = torch.where(
                            is_second_half,
                            torch.full_like(valid_mask, N, dtype=log_y_all.dtype),                 # (B,K,1) float
                            torch.ones_like(valid_mask, dtype=log_y_all.dtype)
                        ) * valid_mask.to(torch.float32)      
                        
            
                        w = torch.zeros_like(log_y_all, dtype=log_y.dtype)
                        w[..., 0] = valid_mask.squeeze(-1)  # mRNA
                        w[..., 1] = tail_mask               # p
                        w[..., 2] = (w_time * valid_mask).squeeze(-1)  # p*
                        w[..., 3] = tail_mask               # R proxy
                        w[..., 4] = tail_mask               # p
                        w[..., 5] = tail_mask               # R proxy
                        w[..., 6] = tail_mask               # p
                        w[..., 7] = tail_mask               # R proxy
                        w[..., 8] = tail_mask               # R proxy
            
                        err2     = (log_pred_all - log_y_all).pow(2) * w
                        mse_chan = err2.sum((0, 1)) / w.sum((0, 1)).clamp_min(1) 
                        mse_chan = torch.cat([mse_chan,extra_mse_chan])                           
                        weighted = self.loss_weight * mse_chan
                        mask = self.loss_weight.ne(0)                       # boolean mask of active terms
                        den  = mask.sum().clamp_min(1).to(weighted.dtype)  # avoid /0
                        val_loss = weighted[mask].sum() / den
                        val_total += val_loss
                        print(weighted[1],weighted[2])

                # keep best by validation
                if val_total + 1e-6 < best_val:
                    best_val = val_total
                    self.best_model = copy.deepcopy(model)
            else:
                # no val split; fall back to train metric for model selection
                if train_total + 1e-6 < best_val:
                    best_val = train_total
                    self.best_model = copy.deepcopy(model)
    
            print(f"Epoch {ep:4d}  train = {train_total:.6f}"
                  + (f" | val = {val_total:.6f}" if val_loader is not None else "")
                  + f" | best_val = {best_val:.6f}")
    
        self.model = self.best_model
        self.model_meta = {
            "in_u": len(self.input_cols) - 1,
            "state_idx": self.ctrl_idx,
            "hidden": hidden,
        }
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"meta": self.model_meta,
                        "state_dict": self.best_model.state_dict()}, p)
            print(f"Model saved to {p}")
            
    @torch.no_grad()
    def plot_parameters(
        self,
        params,
        dt_seq,
        *,
        names=("VTX", "kdm", "VTL", "kmat", 'R', 'lambda', 'kmat_m'),
        save_path="plot/param_traces.png",   # default folder aligned
        dpi=300,
        ):
        
        """ Plot the four ODE parameter traces for one experiment. """
        import matplotlib.pyplot as plt
        from pathlib import Path
    
        # ---------------- normalise input ---------------------------------
        if isinstance(params, dict):
            series = {k: v.squeeze().cpu().numpy() for k, v in params.items()}
        else:
            if params.dim() == 3:                        # (B,K,4) ➜ first batch
                params = params[0]
            series = {names[i]: params[:, i].cpu().numpy()
                      for i in range(params.size(-1))}
    
        # ---------------- build time axis ---------------------------------
        t = torch.cumsum(torch.cat([torch.zeros(1, device=dt_seq.device), dt_seq]), 0)[:-1]
        t = t.cpu().numpy()
    
        # ---------------- plot --------------------------------------------
        fig, ax = plt.subplots(7, 1, figsize=(6, 8), sharex=True)
        for i, name in enumerate(names):
            ax[i].plot(t, series[name])
            ax[i].set_ylabel(name)
            ax[i].grid(alpha=0.3)
        ax[-1].set_xlabel("time [s]")
        fig.tight_layout()
    
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
    
    @torch.no_grad()
    def plot_r2(self,
                pred: list,
                y: list, 
                fname: str,
                save_dir = '/plot/result/',
                ):
        
        def _plot_mapping(y_pred, y_true, r2, fname,save_dir = '',description = ''):
            # Create scatter plot
            fig, _ = plt.subplots(figsize=(6, 4))
            coeffs = np.polyfit(y_true, y_pred, 1)
            line = np.poly1d(coeffs)
            plt.scatter(y_true, y_pred, label='Data points')
            plt.plot(y_true, line(y_true), color='red', label='Best fit line')
            
            # Annotate R^2 on the plot
            plt.text(0.05, 0.95, f'$R^2 = {r2:.3f}$',
                     transform=plt.gca().transAxes,
                     verticalalignment='top',
                     fontsize=12, color = 'r')
            
            # Labels and title
            plt.xlabel(f'Actual Values {description}')
            plt.ylabel(f'Predicted Values {description}')
            plt.tight_layout()    
            plt.title('Seed ' + str(self.choice))            
            fig.savefig(save_dir / f"r2_{fname}.png", dpi=600)
            plt.close(fig)
            
        # The endpoints of the sim and experiment i.e. final yield and peak RNA
        endpoints = list()
        maxima      = list()
        for i in range(len(pred)):
            endpoints.append((pred[i][:,2][-1],y[i][:,2][-1]))
            maxima.append((max(pred[i][:,0]),max(y[i][:,0])))
            
        y_pred,y_true = zip(*endpoints)
        r2 = r2_score(y_true, y_pred)
        fname +=  'p_final'
        _plot_mapping(y_pred, y_true, r2, fname,save_dir = save_dir, description = 'p*')

        y_pred,y_true = zip(*maxima)
        r2 = r2_score(y_true, y_pred)
        fname +=  'mRNA_peak'
        _plot_mapping(y_pred, y_true, r2, fname, save_dir = save_dir, description = 'mRNA (max)')
    
    @torch.no_grad()
    def plot_convergence(
        self,
        *,
        save_dir: str | Path = "plot",     # ← single root folder as requested
        dpi: int = 300,
    ):
        """
        Creates PNGs comparing raw trajectories with model predictions for
        every train/test run and one bar-plot of average MSEs.
        """
        import matplotlib.pyplot as plt
        from pathlib import Path
        from torch.utils.data import DataLoader
        import torch.nn as nn
    
        assert hasattr(self, "model"), "Train a model first."
        assert hasattr(self, "train_idx"), "Call make_train_test_split() first."
        save_dir = Path(save_dir)
        (save_dir / "train").mkdir(parents=True, exist_ok=True)
        (save_dir / "test").mkdir( parents=True, exist_ok=True)
        (save_dir / "result").mkdir( parents=True, exist_ok=True)
        mse_fn = nn.MSELoss(reduction="mean")

        # 1) helper --------------------------------------------------------
        @torch.no_grad()
        def _predict_subset(idx_list):
            self.model.eval()
            ds = self._make_subset_ds(idx_list)
            loader = DataLoader(ds, batch_size=1, num_workers=0)
            preds, targs, lengths, params, dt = [], [], [], [], []
            for x0, u_seq, dna_seq, dt_seq, y_seq in loader:
                x0, u_seq, dna_seq, dt_seq = (
                    x0.to(self.device),
                    u_seq.to(self.device),
                    dna_seq.to(self.device),
                    dt_seq.to(self.device),
                    )
                pred,  p = self.model(x0, u_seq, dna_seq, dt_seq, y_seq=y_seq.to(self.device),
                                      )
                preds.append(pred.squeeze(0).cpu())
                targs.append(y_seq.squeeze(0))
                params.append(p.squeeze(0).cpu())
                dt.append(dt_seq.squeeze(0))
            return preds, targs, params, dt
    
        # 2) run model -----------------------------------------------------
        train_pred, train_y, params_train, dt_train = _predict_subset(self.train_idx)
        test_pred,  test_y, params_test, dt_test = _predict_subset(self.test_idx)
        
        # # 2.5 plot overall results for relevant metrix (MRNA and Protein yield)
        self.plot_r2(train_pred, train_y, 'train', save_dir = save_dir / 'result')
        self.plot_r2(test_pred,  test_y, 'test', save_dir = save_dir / 'result') 
    
        # 3) per-experiment plots -----------------------------------------
        def _plot_runs(pred_list, targ_list, idx_list, subdir):
            for p, y, idx in zip(pred_list, targ_list, idx_list):
                dt = self.dt_list[idx]
                t = torch.cumsum( torch.cat([torch.zeros(1, device=dt.device), dt]), 0)[:-1].cpu()
                p, y = p.numpy(), y.numpy()
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(t, y[:, 0],  label="mRNA (y)")
                ax.plot(t, y[:, 2],  label="p* (y)")
                ax.plot(t, p[:, 0], "--", label="mRNA")
                ax.plot(t, p[:, 1], "--", label="p")
                ax.plot(t, p[:, 2], "--", label="p*")
                ax.set_xlabel("time [s]")
                ax.set_ylabel("RFU")
                ax.set_title(f"Run #{idx}  ({subdir})")
                ax.legend(frameon=False)
                fig.tight_layout()
                fig.savefig(save_dir / subdir / f"run_{idx:03d}.png", dpi=dpi)
                plt.close(fig)
    
        # Plot the data
        _plot_runs(train_pred, train_y, self.train_idx, "train")
        _plot_runs(test_pred,  test_y,  self.test_idx,  "test")
        
        # Plot the parameters over time
        save_dir.mkdir(parents=True, exist_ok=True)

        # -------------- loop with progress print --------------------------
        for i, idx in enumerate(self.train_idx, 1):
            try:
                print(i,len(params_test))
                fname = save_dir / f"run_train_{idx:03d}.png"
                self.plot_parameters(params_train[i], dt_train[i], save_path=fname, dpi=dpi)
            except:
                pass
        for i, idx in enumerate(self.test_idx, 1):
            try:
                print(i,len(params_test))
                fname = save_dir / f"run_test_{idx:03d}.png"
                self.plot_parameters(params_test[i], dt_test[i], save_path=fname, dpi=dpi)
            except:
                pass
            
        # Barplot make a sequence of plots
        train_mse = torch.stack([mse_fn(p, y) for p, y in zip(train_pred, train_y)]).mean()
        test_mse  = torch.stack([mse_fn(p, y) for p, y in zip(test_pred,  test_y )]).mean()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.bar(["Train", "Test"], [train_mse, test_mse])
        ax.set_ylabel("Mean-squared error")
        ax.set_title("Fit quality (lower is better)")
        diff = abs(test_mse - train_mse)
        ax.text(0.5, max(train_mse, test_mse) * 0.9,
                f"Δ = {diff:,.0f}", ha="center")
        fig.tight_layout()
        fig.savefig(save_dir / "mse_comparison.png", dpi=dpi)
        plt.close(fig)
        print(f"✔  All plots saved under “{save_dir.resolve()}”")
        


    

# def make_train_test_split(self, *, test_frac=0.125, val_frac=0.125, num_bins=5):
#     import math
#     assert 0 <= test_frac < 1 and 0 <= val_frac < 1 and test_frac + val_frac < 1
#     N = len(self.y_list)
#     if N == 0:
#         self.train_idx, self.val_idx, self.test_idx = [], [], []
#         return

#     # reproducible per instance, like your original
#     self.choice = random.choice(range(100))
#     rng = random.Random(self.choice)

#     # ---- collect last values of the 3rd column (index 2) for each sequence ----
#     last_vals = []
#     for i, y in enumerate(self.y_list):
#         # y: (K, 3) tensor
#         v = float(y[-1, 2])
#         m = float(y[150, 0])
#         print(m)
#         if not math.isfinite(v):  # guard just in case
#             v = 0.0
#         last_vals.append(v/(m+1))
#     last_vals = np.asarray(last_vals, dtype=float)

#     # ---- build quantile bin edges (balanced bins) ----
#     num_bins = max(1, min(num_bins, N))
#     edges = np.quantile(last_vals, np.linspace(0.0, 1.0, num_bins + 1))
#     # make strictly increasing to keep searchsorted happy
#     eps = 1e-12
#     edges[0] -= eps
#     for j in range(1, edges.size):
#         if edges[j] <= edges[j - 1]:
#             edges[j] = edges[j - 1] + eps

#     # ---- assign each item to a bin ----
#     bin_id = np.searchsorted(edges, last_vals, side="right") - 1
#     bin_id = np.clip(bin_id, 0, num_bins - 1)

#     # ---- per-bin stratified split ----
#     test_idx, val_idx, train_idx = [], [], []
#     for b in range(num_bins):
#         bin_members = [i for i in range(N) if bin_id[i] == b]
#         rng.shuffle(bin_members)

#         n_b = len(bin_members)
#         n_test_b = int(round(n_b * test_frac))
#         n_val_b  = int(round(n_b * val_frac))
#         n_test_b = min(n_test_b, n_b)
#         n_val_b  = min(n_val_b,  n_b - n_test_b)
#         n_train_b = n_b - n_test_b - n_val_b

#         test_idx.extend(bin_members[:n_test_b])
#         val_idx.extend(bin_members[n_test_b:n_test_b + n_val_b])
#         train_idx.extend(bin_members[n_test_b + n_val_b:])

#     # make unique and disjoint (should already be)
#     test_idx = sorted(set(test_idx))
#     val_idx  = sorted(set(val_idx) - set(test_idx))
#     train_idx= sorted(set(train_idx) - set(test_idx) - set(val_idx))

#     self.test_idx, self.val_idx, self.train_idx = test_idx, val_idx, train_idx
#     print(self.test_idx, self.val_idx, self.train_idx)

# helper to assemble a subset dataset
        
# y_clamped   = y_seq.clamp_min(1.0)          # zeros → 1   (B,K,C)
# pred_clamped= pred.clamp_min(1.0)           # ensure log well-defined
# log_y    = torch.log(y_clamped)
# log_pred = torch.log(pred_clamped)

# mask = (torch.arange(y_seq.size(1), device=y_seq.device)
#         [None, :] < lengths[:, None]).unsqueeze(-1)  # (B,K,1)
# err2 = (log_y - log_pred).pow(2) * mask
# valid = mask.sum()
# # mse_chan = err2.sum((0, 1)) / valid          # (3,)
# loss = (self.loss_weight * mse_chan).mean()
# loss.backward()
# nn.utils.clip_grad_norm_(model.parameters(), 5.)
# opt.step()
# total += loss.item()
# ------------------------------------------------------------------
# TRAINING LOOP  (unchanged maths; weight vector now length‑3)
# ------------------------------------------------------------------
# def train(self, *, epochs=200, lr=1e-3,
#                    hidden=128, num_layers=1,
#                    model_cls=None, batch = 300,normalize = False,
#                    save_path: str | Path = None):

#     assert hasattr(self, "train_idx"), "Call make_train_test_split() first."
#     train_ds = self._make_subset_ds(self.train_idx)

#     loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
#                         num_workers=0, collate_fn=collate,
#                         pin_memory=True)

#     model = model_cls(len(self.input_cols) - 1,hidden = hidden, num_layers = num_layers,
#                       ).to(self.device)

#     try:
#         # model = torch.compile(model, mode='max-autotune')  #torch.compile(model, mode="max-autotune", backend="inductor", cudagraphs=True)
#         model = torch.jit.script(model)    
#         print('The model compiled successfully')
#     except AttributeError:
#         print('The model did not compile please check')

#     opt, mse = torch.optim.Adam(model.parameters(), lr=lr), nn.MSELoss()

#     best_val = float('inf')  # initialize with infinity
#     for ep in range(1, epochs + 1):
#         model.train()
#         total = 0
#         for x0, u_seq, dna_seq, dt_seq, y_seq, lengths in loader:
#             x0, u_seq, dna_seq, dt_seq, y_seq, lengths = (
#                 x0.to(self.device), u_seq.to(self.device),
#                 dna_seq.to(self.device), dt_seq.to(self.device),
#                 y_seq.to(self.device), lengths.to(self.device)
#             )
            
#             if normalize:
#                 u_seq = torch.log1p(u_seq)
            
#             # 0) start the opt
#             opt.zero_grad()
#             pred, params = model(x0, u_seq, dna_seq, dt_seq,
#                          y_seq=y_seq)

#             # 1) clamp and log-transofrm
#             y_clamped    = y_seq.clamp_min(1.0)
#             pred_clamped = pred.clamp_min(1.0)
#             log_y, log_pred = torch.log1p(y_clamped), torch.log1p(pred_clamped)  # (B,K,3)
            
#             # 1a) build the extra “R” channel (pred) ---------------------------
#             R_raw   = params[..., 4:5]                       # 5-th column, shape (B,K,1)
#             mean_st = y_seq.mean(dim=-1, keepdim=True)      # mean over the 3 states (B,K,1)
#             R_state = R_raw * mean_st                       # scale by mean state
#             log_R   = torch.log1p(R_state.clamp_min(1.0))     # log-transform
            
#             # 1b) assemble augmented prediction + target tensors ---------------
#             log_pred_all = torch.cat([log_pred, log_R], dim=-1)                    # (B,K,4)
#             log_zero     = torch.zeros_like(log_R)                                 # (B,K,1)
#             log_y_all    = torch.cat([log_y, log_zero], dim=-1)                    # (B,K,4)
            
#             # 1c) assemble non-log data vector + target tensors
#             y_all_abs = torch.cat([y_seq,log_zero], dim=-1)
#             y_pred_abs = torch.cat([pred, R_state], dim=-1)
            
#             # 2) tail-mask for channel-1
#             B, K, _ = y_seq.shape
#             t_grid  = torch.arange(K, device=y_seq.device)           # (K,)
#             valid_mask = (t_grid[None, :] < lengths[:, None]).unsqueeze(-1)   # (B,K,1)
#             cut_idx    = (lengths.float() * 0.95).long()                     # (B,)
#             tail_mask  = (t_grid[None, :] >= cut_idx[:, None])               # (B,K)
#             tail_mask  = tail_mask & valid_mask.squeeze(-1)                  # ensure < lengths
                            
#             # === Extra MSE for last value in column index 1 ===
#             last_indices = (lengths - 1).clamp_min(0)                       # (B,)
#             batch_idx    = torch.arange(B, device=y_seq.device)
#             y_last       = log_y_all[batch_idx, last_indices, 2]            # (B,)
#             pred_last    = log_pred_all[batch_idx, last_indices, 2]         # (B,)
#             extra_mse    = ((pred_last - y_last) ** 2).mean(0, keepdim=True)  # (1,)
            

#             # assemble per-channel weights  (B,K,3)
#             w = torch.zeros_like(log_y_all, dtype=log_y.dtype)
#             w[..., 0] = valid_mask.squeeze(-1)          # mRNA
#             w[..., 2] = valid_mask.squeeze(-1)          # p*
#             w[..., 1] = tail_mask                       # p  (immature)
#             w[..., 3] = tail_mask                       # y (all)

#             # 3) weighted squared error of states
#             err2      = (log_pred_all - log_y_all).pow(2) * w
#             mse_chan  = err2.sum((0,1)) / w.sum((0,1)).clamp_min(1)   # (3,)
#             # err3      = (y_all_abs - y_pred_abs).pow(2) * w
#             # mse_chan_abs = err3.sum((0,1)) / w.sum((0,1)).clamp_min(1)   # (3,)
#             mse_chan = torch.cat([mse_chan, extra_mse], dim=0)               # (5,)
#             loss = (self.loss_weight * mse_chan).mean()
            
#             # 5) combine the losses
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), 1)
#             opt.step()
#             total += loss.item()
        
#         # Save best model
#         if total < best_val - 1e-6:  # improvement
#             best_val = total
#             self.best_model = copy.deepcopy(model)
#         if ep == 1 or ep % 1 == 0:
#             print(f"Epoch {ep:4d}  loss = {total:.6f}, best loss = {best_val}")

#     self.model = self.best_model
#     self.model_meta = {
#         "in_u": len(self.input_cols) - 1,
#         "state_idx": self.ctrl_idx,
#         "hidden": hidden,
#     }
#     if save_path:
#         p = Path(save_path)
#         p.parent.mkdir(parents=True, exist_ok=True)
#         torch.save({"meta": self.model_meta,
#                     "state_dict": self.best_model.state_dict()}, p)
#         print(f"Model saved to {p}")