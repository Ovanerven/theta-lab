from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.data.ode_dataset import ODEDataset, collate
from src.models.ode_rnn import ODERNN
from src.mech.scaffolds import (
    CustomRHSScaffold, 
    rhs_2_torch,
    rhs_3_torch,
    rhs_4_torch,
    rhs_5_torch,
    rhs_6_torch,
    rhs_7_torch,
    rhs_8_torch,
    rhs_9_torch,
    rhs_10_torch,
    rhs_11_torch,
    rhs_12_torch,
    rhs_13_torch
)


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
    log_dir: str = "logs/run"
    save_path: str = "logs/run/model.pt"

    epochs: int = 200
    lr: float = 5e-4
    batch: int = 300
    decay: float = 0.0085
    val_frac: float = 0.15
    seed: int = 42

    hidden: int = 128
    num_layers: int = 1
    grad_clip: float = 1.0

    teacher_forcing: bool = True
    tf_every: int = 50
    tf_drop_epoch: int = 250

    mech_indices_full: Optional[list[int]] = None
    scaffold: str = "reduced5"

    theta_lo: float = 1e-3
    theta_hi: float = 2.0

    n_substeps: int = 1


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_cfg(path: str | Path) -> TrainConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return TrainConfig(**d)


def _select_mech_cols(obs_full: list[int], mech_indices_full: Optional[list[int]]) -> torch.Tensor:
    if mech_indices_full is None:
        return torch.arange(len(obs_full), dtype=torch.long)
    pos = {full_idx: j for j, full_idx in enumerate(obs_full)}
    cols = [pos[i] for i in mech_indices_full]
    return torch.tensor(cols, dtype=torch.long)


def _build_jump(U: int, control_full: list[int], obs_full: list[int], mech_cols: torch.Tensor) -> torch.Tensor:
    mech_full = [obs_full[int(i)] for i in mech_cols.tolist()]
    mech_pos = {full_idx: j for j, full_idx in enumerate(mech_full)}

    jump = torch.zeros(U, len(mech_full), dtype=torch.float32)
    for u, full_idx in enumerate(control_full):
        j = mech_pos.get(full_idx)
        if j is not None:
            jump[u, j] = 1.0
    return jump


def _make_scaffold(name: str) -> CustomRHSScaffold:
    if name == "reduced2":
        return CustomRHSScaffold(P=2, theta_dim=2, rhs_fn=rhs_2_torch)
    if name == "reduced3":
        return CustomRHSScaffold(P=3, theta_dim=4, rhs_fn=rhs_3_torch)
    if name == "reduced4":
        return CustomRHSScaffold(P=4, theta_dim=6, rhs_fn=rhs_4_torch)
    if name == "reduced5":
        return CustomRHSScaffold(P=5, theta_dim=8, rhs_fn=rhs_5_torch)
    if name == "reduced6":
        return CustomRHSScaffold(P=6, theta_dim=10, rhs_fn=rhs_6_torch)
    if name == "reduced7":
        return CustomRHSScaffold(P=7, theta_dim=11, rhs_fn=rhs_7_torch)
    if name == "reduced8":
        return CustomRHSScaffold(P=8, theta_dim=13, rhs_fn=rhs_8_torch)
    if name == "reduced9":
        return CustomRHSScaffold(P=9, theta_dim=14, rhs_fn=rhs_9_torch)
    if name == "reduced10":
        return CustomRHSScaffold(P=10, theta_dim=15, rhs_fn=rhs_10_torch)
    if name == "reduced11":
        return CustomRHSScaffold(P=11, theta_dim=16, rhs_fn=rhs_11_torch)
    if name == "reduced12":
        return CustomRHSScaffold(P=12, theta_dim=17, rhs_fn=rhs_12_torch)
    if name == "full13":
        return CustomRHSScaffold(P=13, theta_dim=18, rhs_fn=rhs_13_torch)
    raise ValueError(f"Unknown scaffold: {name}")


def train(cfg: TrainConfig) -> None:
    start_time = time.time()

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = _device()
    print(f"Using device: {device}")

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
        val_idx, train_idx = np.array([], dtype=int), idx

    train_loader = DataLoader(
        torch.utils.data.Subset(ds, train_idx.tolist()),
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )

    val_loader = None
    if len(val_idx) > 0:
        val_loader = DataLoader(
            torch.utils.data.Subset(ds, val_idx.tolist()),
            batch_size=cfg.batch,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
            pin_memory=True,
        )

    y0_ex, u_ex, _, _ = ds[0]
    U = int(u_ex.shape[1])

    control = ds.control_indices.tolist()
    obs = ds.obs_indices.tolist()

    mech_cols = _select_mech_cols(obs, cfg.mech_indices_full)
    P_mech = int(mech_cols.numel())
    jump = _build_jump(U, control, obs, mech_cols)

    scaffold = _make_scaffold(cfg.scaffold)
    if scaffold.spec.P != P_mech:
        raise ValueError(
            f"scaffold={cfg.scaffold} expects P={scaffold.spec.P}, but mech_indices_full selects P_mech={P_mech}.\n"
            f"Fix by setting mech_indices_full to length {scaffold.spec.P} (or change scaffold)."
        )

    model = ODERNN(
        U=U,
        scaffold=scaffold,
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        u_to_y_jump=jump,
        theta_lo=cfg.theta_lo,
        theta_hi=cfg.theta_hi,
        n_substeps=cfg.n_substeps,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.decay))

    best_val = float("inf")
    best_state = None

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_species_losses: list[np.ndarray] = []

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    mech_cols = mech_cols.to(device)

    for ep in range(1, cfg.epochs + 1):
        ep_start = time.time()
        teacher_forcing = cfg.teacher_forcing and (ep < cfg.tf_drop_epoch)

        model.train()
        train_total = 0.0
        n_batches = 0

        for y0, u_seq, dt_seq, y_seq in train_loader:
            y0 = y0.to(device)[:, mech_cols]
            y_seq = y_seq.to(device)[:, :, mech_cols]
            u_seq = u_seq.to(device)
            dt_seq = dt_seq.to(device)

            opt.zero_grad()
            pred, _ = model(
                y0,
                u_seq,
                dt_seq,
                y_seq=y_seq,
                teacher_forcing=teacher_forcing,
                tf_every=cfg.tf_every,
            )

            loss = loss_fn(pred, y_seq)
            loss.backward()

            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            opt.step()

            train_total += float(loss.item())
            n_batches += 1

        train_loss = train_total / max(1, n_batches)
        train_losses.append(train_loss)

        val_loss = None
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            v_batches = 0
            sp_total = None

            with torch.no_grad():
                for y0, u_seq, dt_seq, y_seq in val_loader:
                    y0 = y0.to(device)[:, mech_cols]
                    y_seq = y_seq.to(device)[:, :, mech_cols]
                    u_seq = u_seq.to(device)
                    dt_seq = dt_seq.to(device)

                    pred, _ = model(y0, u_seq, dt_seq, y_seq=None, teacher_forcing=False)

                    loss = loss_fn(pred, y_seq)
                    val_total += float(loss.item())

                    sp = loss_fn_per_species(pred, y_seq).detach().cpu()
                    sp_total = sp if sp_total is None else sp_total + sp
                    v_batches += 1

            val_loss = val_total / max(1, v_batches)
            val_losses.append(val_loss)

            if sp_total is not None:
                val_species_losses.append((sp_total / max(1, v_batches)).numpy())

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        ep_time = time.time() - ep_start

        if val_loss is None:
            print(f"ep {ep:4d} | train {train_loss:.6f} | {ep_time:.2f}s")
        else:
            sp_str = ""
            if val_species_losses:
                sp = val_species_losses[-1]
                sp_str = "  [" + "  ".join(f"{v:.4f}" for v in sp) + "]"
            print(
                f"ep {ep:4d} | train {train_loss:.6f} | val {val_loss:.6f} | best {best_val:.6f}{sp_str} | {ep_time:.2f}s"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    np.savez(
        log_dir / "loss_curves.npz",
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses) if len(val_losses) > 0 else None,
        val_species_losses=np.array(val_species_losses) if len(val_species_losses) > 0 else None,
        mech_dim=np.int64(P_mech),
        mech_cols=np.array([int(i) for i in mech_cols.detach().cpu().tolist()], dtype=np.int64),
    )
    print(f"Saved loss curves to {log_dir / 'loss_curves.npz'}")

    save_path = Path(cfg.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "best_val": best_val,
            "mech_dim": P_mech,
            "mech_cols": [int(i) for i in mech_cols.detach().cpu().tolist()],
        },
        save_path,
    )
    print(f"Saved best model to {save_path}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f}s ({elapsed/60:.2f}m)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config yaml file")
    parser.add_argument("--no-plot", action="store_true", help="Disable auto plotting after training")
    parser.add_argument("--plot-samples", type=int, default=5, help="How many prediction samples to plot")
    parser.add_argument("--plot-sample-idx", type=int, default=0, help="Which sample idx to use for theta plot")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_cfg(cfg_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("experiments") / f"{timestamp}_{cfg.exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg.log_dir = str(exp_dir / "logs")
    cfg.save_path = str(exp_dir / "model.pt")

    shutil.copy2(cfg_path, exp_dir / "config.yaml")

    print(f"Using config: {cfg_path}")
    print(f"Using experiment structure: experiments/{timestamp}_{cfg.exp_name}/")

    train(cfg)

    if not args.no_plot:
        try:
            from src.scripts.plot_diagnostics import plot_experiment

            plot_experiment(exp_dir, n_samples=int(args.plot_samples), sample_idx=int(args.plot_sample_idx))
        except Exception as e:
            print(f"[plot] failed: {e}")