#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 11:45:30 2025

@author: bob-van-sluijs
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 1) Compute log-uniform distributions of my data
# -----------------------------------------------------------------------------
def _log_uniform(theta_raw, min_val, max_val):
    log_min = torch.log(torch.tensor(min_val))
    log_max = torch.log(torch.tensor(max_val))
    log_theta = log_min + theta_raw * (log_max - log_min)
    return torch.exp(log_theta)

def log_uniform(z, lo, hi, eps=1e-8):
    """Map z∈ℝ⁺ through Softplus then log‑uniform to (lo,hi)."""
    return lo * torch.exp(torch.log(torch.tensor(hi/lo + eps, device=z.device)) * z)



# -----------------------------------------------------------------------------
# 2) Collate/pad the data structs if vectors differ in size
# -----------------------------------------------------------------------------
def collate(batch):
    x0, u_seq, dna_seq, dt_seq, y_seq = zip(*batch)
    lengths   = torch.tensor([seq.size(0) for seq in y_seq])  # ← real K per item
    x0_batch   = torch.stack(x0)                               # (B,2)
    u_padded   = pad_sequence(u_seq,  batch_first=True)        # (B,K,U)
    dna_padded = pad_sequence(dna_seq, batch_first=True)       # (B,K,1)
    dt_padded  = pad_sequence(dt_seq, batch_first=True)        # (B,K)
    y_padded   = pad_sequence(y_seq,  batch_first=True)        # (B,K,2)
    return x0_batch, u_padded, dna_padded, dt_padded, y_padded, lengths


def plot_pred_batch(pred: torch.Tensor,
                    lengths: torch.Tensor | None = None,
                    batch_idx: int | None = None,
                    names: tuple[str, str, str] = ("mRNA pred", "p pred", "p* pred"),
                    figsize: tuple[int, int] = (10, 4)):
    """
    Plot one mini-batch of model predictions.

    Parameters
    ----------
    pred      : Tensor (B, K, 3)
        Model output returned by forward().
    lengths   : Tensor (B,) or None
        Effective sequence lengths; if None, assumes full K for every element.
    batch_idx : int or None
        – None  ➜ plot every sequence in the batch in one figure  
        – n     ➜ plot only that single experiment (index n).
    names     : labels for the three channels.
    figsize   : matplotlib figure size.
    """
    if pred.ndim != 3 or pred.size(-1) != 3:
        raise ValueError("Expect pred with shape (B, K, 3)")

    B, K, _ = pred.shape
    if lengths is None:
        lengths = torch.full((B,), K, dtype=torch.long, device=pred.device)

    batch_idx = 1
    idx_range = range(B) if batch_idx is None else [batch_idx]

    plt.figure(figsize=figsize)
    for b in idx_range:
        t = torch.arange(lengths[b], device=pred.device)
        for ch in range(3):
            alpha = 0.8 if batch_idx is not None else 0.3
            plt.plot(t.detach().cpu().numpy(),
                     pred[b, :lengths[b], ch].detach().cpu().numpy(),
                     label=names[ch] if b == idx_range[0] else None,
                     linewidth=1.6, alpha=1)

    plt.xlabel("time-index k")
    plt.ylabel("concentration (nM)")
    plt.title("Model-predicted trajectories in mini-batch")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_ivtt_batch(y_seq: torch.Tensor,
                    lengths: torch.Tensor | None = None,
                    batch_idx: int | None = None,
                    names: tuple[str,str,str] = ("mRNA", "immature p", "mature p*"),
                    figsize: tuple[int,int] = (10, 4)):
    """
    Visualise y_seq for one mini-batch.

    Parameters
    ----------
    y_seq   : Tensor (B, K, 3)
        Ground-truth states returned by your DataLoader.
    lengths : Tensor (B,) or None
        Effective sequence length of each experiment; if None, assumes
        full K for every batch element.
    batch_idx : int or None
        If given, plot only that element of the batch; else plot all B
        experiments with light alpha.
    names  : tuple(str,str,str)
        Legend labels for the three channels.
    """
    if y_seq.ndim != 3 or y_seq.size(-1) != 3:
        raise ValueError("Expect y_seq with shape (B, K, 3)")

    B, K, _ = y_seq.shape
    if lengths is None:
        lengths = torch.full((B,), K, dtype=torch.long, device=y_seq.device)

    batch_idx = 4
    # pick subset
    idx_range = range(B) if batch_idx is None else [batch_idx]

    # plot
    plt.figure(figsize=figsize)
    for b in idx_range:
        t = torch.arange(lengths[b], device=y_seq.device)
        for ch in range(3):
            alpha = 0.8 if batch_idx is not None else 0.3
            plt.plot(t.cpu(), y_seq[b, :lengths[b], ch].cpu(),
                     label=names[ch] if b == idx_range[0] else None,
                     linewidth=1.6, alpha=1)

    plt.xlabel("time-index k")
    plt.ylabel("concentration (nM)")
    plt.title("Ground-truth trajectories in mini-batch")
    plt.legend()
    plt.tight_layout()
    plt.show()



