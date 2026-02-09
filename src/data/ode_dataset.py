from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

class ODEDataset(Dataset):
    """
    Expects .npz with:
      y0   (N,P)    observed state at t0
      u_seq(N,K,U)  control inputs
      y_seq(N,K,P)  observed trajectories
      t_obs(K+1,)   shared time grid
    """

    def __init__(self, npz_path: str | Path):
        data = np.load(str(npz_path), allow_pickle=False)

        self.y0 = data["y0"].astype(np.float32)        
        self.u_seq = data["u_seq"].astype(np.float32)
        self.y_seq = data["y_seq"].astype(np.float32)
        self.t_obs = data["t_obs"].astype(np.float32)
        self.control_indices = data["control_indices"].astype(np.int64)
        self.obs_indices = data["obs_indices"].astype(np.int64)
        
        # Compute dt_seq from t_obs (K+1,) -> (K,)
        self.dt = np.diff(self.t_obs).astype(np.float32)

        assert self.y0.ndim == 2
        assert self.u_seq.ndim == 3
        assert self.y_seq.ndim == 3
        assert self.t_obs.ndim == 1

        N = self.y0.shape[0]
        K = self.u_seq.shape[1]
        assert self.u_seq.shape[0] == N and self.y_seq.shape[0] == N
        assert self.t_obs.shape[0] == K + 1

    def __len__(self) -> int:
        return int(self.y0.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.y0[idx]),     # (P,) initial observed state y0
            torch.from_numpy(self.u_seq[idx]),  # (K,U) control inputs
            torch.from_numpy(self.dt),          # (K,) time intervals (shared across samples)
            torch.from_numpy(self.y_seq[idx]),  # (K,P) observed trajectory
        )


def collate(batch):
    y0, u_seq, dt_seq, y_seq = zip(*batch)
    return (
        torch.stack(y0, dim=0),
        torch.stack(u_seq, dim=0),
        torch.stack(dt_seq, dim=0),
        torch.stack(y_seq, dim=0),
    )
