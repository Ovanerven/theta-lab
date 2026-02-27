# train.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

from scaffolds import SCAFFOLDS
from ode_rnn import ODERNN
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

    def __len__(self) -> int:
        return self.y0.shape[0]

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(self.y0[i]),  # (P_obs,)
            torch.from_numpy(self.u_seq[i]),  # (K,U)
            torch.from_numpy(self.y_seq[i]),  # (K,P_obs)
        )


def collate(batch):
    y0, u, y = zip(*batch)
    return torch.stack(y0), torch.stack(u), torch.stack(y)


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


def loss_fn_per_species(pred: torch.Tensor, y_seq: torch.Tensor) -> torch.Tensor:
    log_y = torch.log1p(y_seq)
    log_pred = torch.log1p(pred)
    return (log_pred - log_y).pow(2).mean(dim=(0, 1))


@dataclass
class TrainConfig:
    dataset_path: str

    exp_name: str = "run"
    out_root: str = "experiments"

    save_model_name: str = "model.pt"  # saved in exp_dir/
    save_last_name: str = "model_last.pt"  # saved in exp_dir/
    save_curves_name: str = "loss_curves.npz"  # saved in exp_dir/logs/

    epochs: int = 200
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 0.0
    val_frac: float = 0.15
    seed: int = 42

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

    grad_clip: float = 1.0
    teacher_forcing: bool = True
    tf_every: int = 50
    tf_drop_epoch: int = 10**9

    # checkpointing cadence (0 disables periodic ckpts)
    ckpt_every: int = 10


def load_cfg(path: str | Path) -> TrainConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return TrainConfig(**d)


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

    device = device_auto()
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(cfg.out_root) / f"{timestamp}_{cfg.exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = exp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config for later reconstruction
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(cfg.__dict__, sort_keys=False))
    print(f"Experiment: {exp_dir}")

    ds = ODEDataset(cfg.dataset_path)
    N = len(ds)

    idx = np.arange(N)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(idx)

    if cfg.val_frac > 0.0 and N > 1:
        n_val = max(1, int(N * cfg.val_frac))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
    else:
        val_idx = np.array([], dtype=int)
        train_idx = idx

    train_loader = DataLoader(
        torch.utils.data.Subset(ds, train_idx.tolist()),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=bool(cfg.pin_memory),
    )

    val_loader = None
    if len(val_idx) > 0:
        val_loader = DataLoader(
            torch.utils.data.Subset(ds, val_idx.tolist()),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate,
            pin_memory=bool(cfg.pin_memory),
        )

    # infer dims
    y0_ex, u_ex, _ = ds[0]
    P_obs = int(y0_ex.shape[0])
    U = int(u_ex.shape[-1])

    if cfg.scaffold not in SCAFFOLDS:
        raise ValueError(f"Unknown scaffold '{cfg.scaffold}'. Available: {list(SCAFFOLDS.keys())}")
    scaffold = SCAFFOLDS[cfg.scaffold]

    if scaffold.P != P_obs:
        raise ValueError(f"Scaffold {cfg.scaffold} expects P={scaffold.P}, but dataset has P_obs={P_obs}.")

    u_to_y_jump = make_u_to_y_jump(ds.control_indices, ds.obs_indices, device=device)  # (U,P_obs)

    model = ODERNN(
        U=U,
        scaffold=scaffold,
        u_to_y_jump=u_to_y_jump,
        hidden=cfg.hidden,
        lift_dim=cfg.lift_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        theta_lo=cfg.theta_lo,
        theta_hi=cfg.theta_hi,
        n_substeps=cfg.n_substeps,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    mech_names = ds.obs_names.tolist() if ds.obs_names is not None else None

    print(f"Data: N={N} | train={len(train_idx)} | val={len(val_idx)}")
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

    for ep in range(1, cfg.epochs + 1):
        ep_t0 = time.time()
        teacher_forcing = bool(cfg.teacher_forcing) and (ep < int(cfg.tf_drop_epoch))

        # ---- train
        model.train()
        tr_total = 0.0
        tr_batches = 0

        for y0, u_seq, y_seq in train_loader:
            dt_seq = torch.from_numpy(ds.dt)
            dt_seq = dt_seq[None, :].expand(y0.shape[0], -1)

            y0 = y0.to(device)
            y_seq = y_seq.to(device)
            u_seq = u_seq.to(device)
            dt_seq = dt_seq.to(device)

            opt.zero_grad(set_to_none=True)
            pred, _ = model(
                y0,
                u_seq,
                dt_seq,
                y_seq=y_seq,
                teacher_forcing=teacher_forcing,
                tf_every=int(cfg.tf_every),
            )
            loss = loss_fn(pred, y_seq)
            loss.backward()

            if cfg.grad_clip and float(cfg.grad_clip) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))

            opt.step()

            tr_total += float(loss.item())
            tr_batches += 1

        tr_loss = tr_total / max(1, tr_batches)
        train_losses.append(tr_loss)

        # ---- val
        va_loss = None
        sp_last = None

        if val_loader is not None:
            model.eval()
            va_total = 0.0
            va_batches = 0
            sp_total = None

            with torch.no_grad():
                for y0, u_seq, y_seq in val_loader:
                    dt_seq = torch.from_numpy(ds.dt)
                    dt_seq = dt_seq[None, :].expand(y0.shape[0], -1)

                    y0 = y0.to(device)
                    y_seq = y_seq.to(device)
                    u_seq = u_seq.to(device)
                    dt_seq = dt_seq.to(device)

                    pred, _ = model(y0, u_seq, dt_seq, y_seq=None, teacher_forcing=False)

                    loss = loss_fn(pred, y_seq)
                    va_total += float(loss.item())

                    sp = loss_fn_per_species(pred, y_seq).detach().cpu()
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

    # save best model (plot expects exp_dir/model.pt)
    save_path = exp_dir / cfg.save_model_name
    torch.save(
        {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "best_val": float(best_val)},
        save_path,
    )
    print(f"Saved best model to {save_path}")

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.2f}s ({elapsed/60:.2f}m)")

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-samples", type=int, default=5)
    parser.add_argument("--plot-sample-idx", type=int, default=0)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    train(cfg, no_plot=bool(args.no_plot), plot_samples=int(args.plot_samples), plot_sample_idx=int(args.plot_sample_idx))