from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from src.data.ode_dataset import ODEDataset, collate
from src.models.ode_rnn import ODERNN
from src.mech.scaffolds import make_scaffold


@dataclass
class PlotConfig:
    n_samples: int = 5
    sample_idx: int = 0


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


def _rebuild_model_from_experiment(
    exp_dir: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, ODEDataset, torch.Tensor, list[str], list[str]]:
    cfg = _load_yaml(exp_dir / "config.yaml")

    dataset_path = Path(cfg["dataset_path"])
    ds = ODEDataset(dataset_path)

    control = ds.control_indices.tolist()
    obs = ds.obs_indices.tolist()

    mech_indices_full = cfg.get("mech_indices_full", None)
    mech_cols = _select_mech_cols(obs, mech_indices_full)
    P_mech = int(mech_cols.numel())

    y0_ex, u_ex, _, _ = ds[0]
    U = int(u_ex.shape[1])

    jump = _build_jump(U, control, obs, mech_cols)

    scaffold_name = cfg.get("scaffold", "reduced5")
    scaffold = make_scaffold(scaffold_name)

    if scaffold.spec.P != P_mech:
        raise ValueError(
            f"scaffold={scaffold_name} expects P={scaffold.spec.P}, but mech selection gives P_mech={P_mech}"
        )

    state_names = scaffold.state_names()
    param_names = scaffold.param_names()

    model = ODERNN(
        U=U,
        scaffold=scaffold,
        hidden=int(cfg.get("hidden", 128)),
        num_layers=int(cfg.get("num_layers", 1)),
        u_to_y_jump=jump,
        theta_lo=float(cfg.get("theta_lo", 1e-3)),
        theta_hi=float(cfg.get("theta_hi", 2.0)),
        n_substeps=int(cfg.get("n_substeps", 1)),
    ).to(device)

    ckpt = torch.load(exp_dir / "model.pt", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, ds, mech_cols, state_names, param_names


def _plot_loss_curves(loss_npz: Path, out_dir: Path):
    data = np.load(loss_npz, allow_pickle=True)
    train_losses = data["train_losses"]
    val_losses = data["val_losses"] if "val_losses" in data.files and data["val_losses"] is not None else None

    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, label="train", linewidth=2)
    if val_losses is not None and len(val_losses) > 0:
        ax.plot(epochs, val_losses, label="val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log1p space)")
    ax.set_title("Loss curves")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close(fig)


def _plot_val_species_losses(loss_npz: Path, out_dir: Path, state_names: list[str]):
    data = np.load(loss_npz, allow_pickle=True)
    if "val_species_losses" not in data.files or data["val_species_losses"] is None:
        return
    V = data["val_species_losses"]
    if V.size == 0:
        return

    # heatmap V: (E, P)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(V.T, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Species")
    ax.set_title("Val per-species loss (log1p-MSE)")
    ax.set_yticks(np.arange(len(state_names)))
    ax.set_yticklabels(state_names)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "val_species_heatmap.png", dpi=150)
    plt.close(fig)

    # final epoch bar
    last = V[-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(last)), last)
    ax.set_xticks(np.arange(len(state_names)))
    ax.set_xticklabels(state_names, rotation=0)
    ax.set_xlabel("Species")
    ax.set_ylabel("Val loss")
    ax.set_title("Val per-species loss (final epoch)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "val_species_final.png", dpi=150)
    plt.close(fig)


def _plot_predictions(
    model: torch.nn.Module,
    ds: ODEDataset,
    mech_cols: torch.Tensor,
    state_names: list[str],
    out_dir: Path,
    n_samples: int,
    device: torch.device,
):
    n_samples = min(int(n_samples), len(ds))
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=n_samples,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    y0, u_seq, dt_seq, y_seq = next(iter(loader))

    mech_cols_dev = mech_cols.to(device)
    y0 = y0.to(device)[:, mech_cols_dev]
    y_seq = y_seq.to(device)[:, :, mech_cols_dev]
    u_seq = u_seq.to(device)
    dt_seq = dt_seq.to(device)

    with torch.no_grad():
        pred, _ = model(y0, u_seq, dt_seq, y_seq=None, teacher_forcing=False)

    y_true = y_seq.cpu().numpy()
    y_pred = pred.cpu().numpy()
    dt_np = dt_seq.cpu().numpy()

    P = y_pred.shape[-1]

    for i in range(n_samples):
        t = np.concatenate([[0.0], np.cumsum(dt_np[i])])
        fig, axes = plt.subplots(P, 1, figsize=(11, max(6, 2.0 * P)), sharex=True)
        if P == 1:
            axes = [axes]

        for p, ax in enumerate(axes):
            ax.plot(t[1:], y_true[i, :, p], linewidth=2, label="true")
            ax.plot(t[1:], y_pred[i, :, p], linewidth=2, linestyle="--", label="pred")
            ax.set_ylabel(state_names[p] if p < len(state_names) else f"s{p}")
            ax.grid(True, alpha=0.25)
            if p == 0:
                ax.legend()

        axes[-1].set_xlabel("Time")
        fig.suptitle(f"Prediction vs truth (sample {i})")
        fig.tight_layout()
        fig.savefig(out_dir / f"pred_vs_true_{i:03d}.png", dpi=150)
        plt.close(fig)


def _plot_theta(
    model: torch.nn.Module,
    ds: ODEDataset,
    mech_cols: torch.Tensor,
    param_names: list[str],
    out_dir: Path,
    sample_idx: int,
    device: torch.device,
):
    sample_idx = int(sample_idx)
    y0, u_seq, dt_seq, _ = ds[sample_idx]

    mech_cols_dev = mech_cols.to(device)
    y0 = y0.unsqueeze(0).to(device)[:, mech_cols_dev]
    u_seq = u_seq.unsqueeze(0).to(device)
    dt_seq = dt_seq.unsqueeze(0).to(device)

    with torch.no_grad():
        _, theta = model(y0, u_seq, dt_seq, y_seq=None, teacher_forcing=False)

    theta_np = theta[0].cpu().numpy()  # (K, theta_dim)
    dt = dt_seq[0].cpu().numpy()
    t = np.concatenate([[0.0], np.cumsum(dt)])
    tt = t[1:]

    _, D = theta_np.shape
    n_cols = 2
    n_rows = (D + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.2 * n_rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for j, ax in enumerate(axes):
        if j < D:
            ax.plot(tt, theta_np[:, j], linewidth=1.8)
            name = param_names[j] if j < len(param_names) else f"θ{j}"
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.25)
        else:
            ax.axis("off")

    axes[min(D - 1, len(axes) - 1)].set_xlabel("Time")
    fig.suptitle(f"Learned θ(t) (sample {sample_idx})")
    fig.tight_layout()
    fig.savefig(out_dir / f"theta_sample{sample_idx}.png", dpi=150)
    plt.close(fig)


def plot_experiment(exp_dir: str | Path, n_samples: int = 5, sample_idx: int = 0) -> Path:
    exp_dir = Path(exp_dir)
    out_dir = exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _device()

    model, ds, mech_cols, state_names, param_names = _rebuild_model_from_experiment(exp_dir, device=device)

    loss_npz = exp_dir / "logs" / "loss_curves.npz"
    if loss_npz.exists():
        _plot_loss_curves(loss_npz, out_dir)
        _plot_val_species_losses(loss_npz, out_dir, state_names=state_names)

    _plot_predictions(
        model,
        ds,
        mech_cols,
        state_names=state_names,
        out_dir=out_dir,
        n_samples=n_samples,
        device=device,
    )
    _plot_theta(
        model,
        ds,
        mech_cols,
        param_names=param_names,
        out_dir=out_dir,
        sample_idx=sample_idx,
        device=device,
    )

    print(f"Saved plots to {out_dir}")
    return out_dir


if __name__ == "__main__":
    import sys

    # usage: python -m src.scripts.plot_diagnostics experiments/<timestamp>_<name> [n_samples] [sample_idx]
    exp = sys.argv[1] if len(sys.argv) > 1 else "experiments/latest"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    s = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    plot_experiment(exp, n_samples=n, sample_idx=s)