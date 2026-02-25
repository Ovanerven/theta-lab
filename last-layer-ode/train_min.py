from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scaffold_min import SCAFFOLDS
from ode_rnn import ODERNN, make_u_to_y_jump


class ODEDataset(Dataset):
    def __init__(self, npz_path: str | Path):
        d = np.load(str(npz_path))
        self.y0 = d["y0"].astype(np.float32)           # (N,P)
        self.u_seq = d["u_seq"].astype(np.float32)     # (N,K,U)
        self.y_seq = d["y_seq"].astype(np.float32)     # (N,K,P)
        t_obs = d["t_obs"].astype(np.float32)          # (K+1,)
        self.dt = np.diff(t_obs).astype(np.float32)    # (K,)

    def __len__(self) -> int:
        return self.y0.shape[0]

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(self.y0[i]),
            torch.from_numpy(self.u_seq[i]),
            torch.from_numpy(self.dt),
            torch.from_numpy(self.y_seq[i]),
        )


def collate(batch):
    y0, u, dt, y = zip(*batch)
    return torch.stack(y0), torch.stack(u), torch.stack(dt), torch.stack(y)


def loss_fn(pred: torch.Tensor, y_seq: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss in log10 space."""
    # Edit on 13/02: based on Bob's feedback, looked to fix the loss. Stopped clamping loss at 1
    # eps = 1e-8 # small constant to avoid log(0) in the loss.
    # log_y = torch.log10(y_seq + eps)
    # log_pred = torch.log10(pred + eps)

    # New proposal: we do log1p with no clampmin at 1. But does this drown out boluses?
    log_y = torch.log1p(y_seq)
    log_pred = torch.log1p(pred)
    # log_y = y_seq
    # log_pred = pred

    # Old idea: clamp_min at 1, use log1p.
    # y_clamped = y_seq.clamp_min(1.0)
    # pred_clamped = pred.clamp_min(1.0)
    # log_y = torch.log1p(y_clamped)
    # log_pred = torch.log1p(pred_clamped)

    return (log_pred - log_y).pow(2).mean()  # MSE


@dataclass
class Config:
    dataset: str
    scaffold: str = "reduced5"
    epochs: int = 50
    batch_size: int = 128
    lr: float = 5e-4
    hidden: int = 128
    lift_dim: int = 32
    tf_every: int = 50


def main(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ODEDataset(cfg.dataset)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)

    # infer dims from data + scaffold
    y0_ex, u_ex, _, _ = ds[0]
    U = u_ex.shape[-1]
    sc = SCAFFOLDS[cfg.scaffold]
    P = sc.P

    # simplest “jump”: identity from controls to observed dims if U==P
    # (replace this with your real mapping if controls are a subset of full state)
    if U != P:
        u_to_y_jump = make_u_to_y_jump(list(range(U)), list(range(P)))

    u_to_y_jump = torch.eye(U, P)

    model = ODERNN(
        U=U,
        scaffold=sc,
        u_to_y_jump=u_to_y_jump,
        hidden=cfg.hidden,
        lift_dim=cfg.lift_dim,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for ep in range(cfg.epochs):
        model.train()
        running = 0.0
        for y0, u_seq, dt_seq, y_seq in dl:
            y0, u_seq, dt_seq, y_seq = y0.to(device), u_seq.to(device), dt_seq.to(device), y_seq.to(device)

            pred, theta = model(y0, u_seq, dt_seq, y_seq=y_seq, teacher_forcing=True, tf_every=cfg.tf_every)
            loss = loss_fn(pred, y_seq)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += float(loss.item())

        print(f"epoch {ep:03d} | loss {running / len(dl):.6f}")


if __name__ == "__main__":
    # quick-and-dirty config (keep it simple on purpose)
    cfg = Config(dataset="path/to/dataset.npz")
    main(cfg)