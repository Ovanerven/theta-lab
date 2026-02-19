"""
Plotting utilities for KineticsRNN training diagnostics.

Public API
----------
build_model_from_checkpoint(ckpt_path, dataset_path, obs_indices=None, device="cpu")
    -> (KineticsRNN, ODEDataset)

plot_loss_curves(loss_file, out_dir)
plot_predictions(model_path, dataset_path, out_dir, n_samples=5, device=None)
plot_learned_parameters(model_path, dataset_path, sample_idx=0, output_path=None)
plot_evolution(log_dir, dataset_path, sample_idx=0, out_dir=None)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.sim.mechanisms import MECH as MECH_REGISTRY
from src.models.kinetics_rnn import KineticsRNN
from src.data.ode_dataset import ODEDataset, collate


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model_from_checkpoint(
    ckpt_path: Path | str,
    dataset_path: Path | str,
    obs_indices: Optional[List[int]] = None,
    device: str | torch.device = "cpu",
) -> Tuple[KineticsRNN, ODEDataset]:
    """
    Reconstruct a KineticsRNN from a saved checkpoint + dataset.

    Parameters
    ----------
    ckpt_path : path to the .pt checkpoint file
    dataset_path : path to the .npz dataset file
    obs_indices : column selection passed to ODEDataset (which columns of
        y0/y_seq to use).  Usually you want this to match what was used during
        training.  If None, uses all species stored in the file.
    device : torch device string or object
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt["state_dict"]

    # --- Mechanism ---
    mech_name = ckpt.get("mechanism")
    if mech_name is None:
        # Fallback: guess from head size
        n_param = next(v for k, v in state_dict.items() if "head.weight" in k).shape[0]
        mech_name = {8: "reduced5", 19: "full13"}.get(n_param)
        if mech_name is None:
            raise ValueError(
                f"Cannot infer mechanism from checkpoint (head size={n_param}). "
                "Please retrain with the new train.py."
            )
    mech = MECH_REGISTRY[mech_name]

    # --- obs_indices stored in checkpoint (full-state indices) ---
    ckpt_obs = ckpt.get("obs_indices")

    # --- Dataset: load with the same column selection used during training ---
    ds = ODEDataset(str(dataset_path), obs_indices=obs_indices if obs_indices is not None else ckpt_obs)
    y0_ex, u_ex, _, _ = ds[0]
    P = int(y0_ex.shape[0])
    U = int(u_ex.shape[1])

    # Resolve final obs_indices (checkpoint takes priority)
    if ckpt_obs is not None:
        obs_indices = ckpt_obs
    else:
        obs_indices = ds.obs_indices.tolist()

    # --- Build jump matrix ---
    ctrl    = ds.control_indices.tolist()
    obs_pos = {full_idx: p for p, full_idx in enumerate(obs_indices)}
    jump    = torch.zeros(U, P, dtype=torch.float32)
    for u_i, full_idx in enumerate(ctrl):
        p = obs_pos.get(full_idx)
        if p is not None:
            jump[u_i, p] = 1.0

    # --- Training cfg ---
    train_cfg = {}
    raw_cfg = ckpt.get("cfg", {})
    if isinstance(raw_cfg, dict):
        train_cfg = raw_cfg.get("train", raw_cfg)  # handle nested or flat

    hidden     = int(train_cfg.get("hidden", 128))
    num_layers = int(train_cfg.get("num_layers", 1))

    model = KineticsRNN(
        mech=mech,
        obs_indices=obs_indices,
        in_u=U,
        u_to_y_jump=jump,
        hidden=hidden,
        num_layers=num_layers,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, ds


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    loss_file: Path,
    out_dir: Path,
):
    """Plot training and validation loss curves from a loss_curves.npz file."""
    data = np.load(loss_file)
    train_losses = data["train_losses"]
    val_losses   = data.get("val_losses")

    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(epochs, train_losses, label="train", linewidth=2)
    if val_losses is not None and len(val_losses) > 0:
        ax.plot(epochs, val_losses, label="val", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE in log10 space", fontsize=12)
    ax.set_title("Training and Validation Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = Path(out_dir) / "loss_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss curve plot: {out_path}")


# ---------------------------------------------------------------------------
# Prediction vs truth
# ---------------------------------------------------------------------------

def plot_predictions(
    model_path: Path,
    dataset_path: Path,
    out_dir: Path,
    n_samples: int = 5,
    device: Optional[torch.device] = None,
):
    """Plot predicted vs ground-truth trajectories for *n_samples* samples."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model, ds = build_model_from_checkpoint(model_path, dataset_path, device=device)
    P = model.P

    loader = DataLoader(
        ds,
        batch_size=min(n_samples, len(ds)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    y0_batch, u_batch, dt_batch, y_batch = next(iter(loader))
    y0_batch  = y0_batch.to(device)
    u_batch   = u_batch.to(device)
    dt_batch  = dt_batch.to(device)
    y_batch   = y_batch.to(device)

    with torch.no_grad():
        pred_batch, _ = model(y0_batch, u_batch, dt_batch, y_seq=None, teacher_forcing=False)

    species_names = MECH_REGISTRY[model.mech_name].state_names if hasattr(model, "mech_name") else [f"S{i}" for i in range(P)]

    for idx in range(min(n_samples, y0_batch.shape[0])):
        dt_np  = dt_batch[idx].cpu().numpy()
        t      = np.concatenate([[0.0], np.cumsum(dt_np)])
        y_true = y_batch[idx].cpu().numpy()
        y_pred = pred_batch[idx].cpu().numpy()

        fig, axes = plt.subplots(P, 1, figsize=(12, max(8, P * 0.8)), sharex=True)
        if P == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(t[1:], y_true[:, i], label="true",  linewidth=2, alpha=0.8)
            ax.plot(t[1:], y_pred[:, i], label="pred",  linewidth=2, linestyle="--", alpha=0.8)
            ax.set_ylabel(species_names[i], fontsize=11)
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.legend(loc="best", fontsize=10)

        axes[-1].set_xlabel("Time (s)", fontsize=12)
        fig.suptitle(f"Prediction vs Truth (Sample {idx})", fontsize=14)
        fig.tight_layout()

        out_path = Path(out_dir) / f"pred_vs_true_sample{idx:03d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {n_samples} prediction plots to {out_dir}")


# ---------------------------------------------------------------------------
# Learned parameters
# ---------------------------------------------------------------------------

def plot_learned_parameters(
    model_path: str,
    dataset_path: str,
    sample_idx: int = 0,
    output_path: str = None,
):
    """
    Load a trained model, run inference on one sample, and plot θ(t).

    Parameters
    ----------
    model_path : path to .pt checkpoint
    dataset_path : path to .npz dataset
    sample_idx : which sample to visualise
    output_path : where to save the PNG; defaults to model directory
    """
    model, ds = build_model_from_checkpoint(model_path, dataset_path)

    if sample_idx >= len(ds):
        raise ValueError(f"sample_idx {sample_idx} out of range (dataset has {len(ds)} samples)")

    y0, u_seq, dt_seq, y_seq = ds[sample_idx]

    with torch.no_grad():
        y0_batch = y0.unsqueeze(0)
        u_batch  = u_seq.unsqueeze(0)
        dt_batch = dt_seq.unsqueeze(0)
        pred, theta = model(y0_batch, u_batch, dt_batch, y_seq=None, teacher_forcing=False)

    theta_np  = theta[0].cpu().numpy()        # (K, n_params)
    dt_np     = dt_seq.cpu().numpy()
    t         = np.concatenate([[0.0], np.cumsum(dt_np)])
    t_theta   = t[1:]

    n_params    = theta_np.shape[1]
    param_names = MECH_REGISTRY[model.mech_name].param_names if hasattr(model, "mech_name") else [f"θ{i+1}" for i in range(n_params)]

    ncols = min(3, n_params)
    nrows = (n_params + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2 * nrows), sharex=True)
    axes = np.array(axes).flatten()

    for i, (ax, name) in enumerate(zip(axes[:n_params], param_names)):
        ax.plot(t_theta, theta_np[:, i], linewidth=1.5, color=f"C{i % 10}")
        ax.set_ylabel(name, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    for i in range(n_params, len(axes)):
        axes[i].axis("off")

    for i in range(nrows * ncols - ncols, nrows * ncols):
        if i < len(axes):
            axes[i].set_xlabel("Time", fontsize=12)

    model_name = Path(model_path).stem
    fig.suptitle(f"Learned Time-Varying Parameters θ(t)\n{model_name}  –  Sample {sample_idx}", fontsize=14, y=0.998)
    fig.tight_layout()

    if output_path is None:
        output_path = Path(model_path).parent / f"theta_{model_name}_sample{sample_idx}.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved parameter plot to {output_path}")
    plt.close(fig)

    return theta_np, t_theta


# ---------------------------------------------------------------------------
# Epoch evolution
# ---------------------------------------------------------------------------

def _load_epoch_checkpoint(ckpt_path: Path, device: str, dataset_path: str):
    """Load model + dataset + epoch number from a periodic checkpoint."""
    model, ds = build_model_from_checkpoint(ckpt_path, dataset_path, device=device)
    ckpt  = torch.load(str(ckpt_path), map_location="cpu")
    epoch = ckpt.get("epoch", 0)
    return model, ds, epoch


def _predict_sample(model, y0_np, u_seq_np, dt_seq_np, device):
    """Run model inference on a single numpy sample."""
    y0_t  = torch.from_numpy(y0_np).float().unsqueeze(0).to(device)
    u_t   = torch.from_numpy(u_seq_np).float().unsqueeze(0).to(device)
    dt_t  = torch.from_numpy(dt_seq_np).float().unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred, _ = model(y0_t, u_t, dt_t, y_seq=None, teacher_forcing=False)

    return y_pred.squeeze(0).cpu().numpy()


def plot_evolution(
    log_dir: str,
    dataset_path: str,
    sample_idx: int = 0,
    out_dir: str = None,
):
    """
    Plot how predictions evolve across training epochs by loading
    periodic checkpoints saved in *log_dir*/checkpoints/.

    Parameters
    ----------
    log_dir : path to training log directory (must contain checkpoints/)
    dataset_path : path to .npz dataset
    sample_idx : which sample to visualise
    out_dir : where to save the PNG; defaults to log_dir
    """
    log_dir  = Path(log_dir)
    ckpt_dir = log_dir / "checkpoints"

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found at {ckpt_dir}")

    ckpt_paths = sorted(ckpt_dir.glob("model_ep*.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

    print(f"Found {len(ckpt_paths)} checkpoints")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    first_model, dataset, _ = _load_epoch_checkpoint(ckpt_paths[0], device, dataset_path)
    y0, u_seq, dt_seq, y_true = dataset[sample_idx]

    y0_np     = y0.numpy()
    u_seq_np  = u_seq.numpy()
    dt_seq_np = dt_seq.numpy()
    y_true_np = y_true.numpy()
    t_obs     = np.concatenate([[0], np.cumsum(dt_seq_np)])

    epochs      = []
    predictions = []
    for ckpt_path in ckpt_paths:
        model, _, epoch = _load_epoch_checkpoint(ckpt_path, device, dataset_path)
        predictions.append(_predict_sample(model, y0_np, u_seq_np, dt_seq_np, device))
        epochs.append(epoch)

    n_species       = y_true_np.shape[1]
    species_names = MECH_REGISTRY[first_model.mech_name].state_names if hasattr(first_model, "mech_name") else [f"S{i}" for i in range(n_species)]

    fig, axes = plt.subplots(n_species, 1, figsize=(12, max(8, n_species * 0.8)), sharex=True)
    if n_species == 1:
        axes = [axes]

    cmap = plt.cm.viridis
    for i, ax in enumerate(axes):
        ax.plot(t_obs[1:], y_true_np[:, i], "k-", linewidth=2, label="Truth", alpha=0.8)

        for j, (epoch, y_pred) in enumerate(zip(epochs, predictions)):
            color = cmap(j / max(len(epochs) - 1, 1))
            alpha = 0.3 + 0.5 * (j / max(len(epochs) - 1, 1))
            label = f"Epoch {epoch}" if j % max(1, len(epochs) // 5) == 0 else None
            ax.plot(t_obs[1:], y_pred[:, i], color=color, alpha=alpha, linewidth=1, label=label)

        ax.set_ylabel(f"{species_names[i]} concentration", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time", fontsize=12)
    fig.suptitle(
        f"Prediction Evolution Over Training (Sample {sample_idx})\n"
        f"Epochs: {epochs[0]} → {epochs[-1]}",
        fontsize=14,
    )
    plt.tight_layout()

    if out_dir is None:
        out_dir = log_dir
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"epoch_evolution_sample{sample_idx}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved epoch evolution plot to {out_path}")
    plt.close()
