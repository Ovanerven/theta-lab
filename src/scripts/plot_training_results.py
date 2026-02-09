from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.simple_ode_rnn import SimpleRNN
from src.data.ode_dataset import ODEDataset, collate


def plot_loss_curves(
    loss_file: Path,
    out_dir: Path,
):
    """Plot training and validation loss curves."""
    data = np.load(loss_file)
    train_losses = data["train_losses"]
    val_losses = data.get("val_losses")

    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="train", linewidth=2)
    if val_losses is not None and len(val_losses) > 0:
        ax.plot(epochs, val_losses, label="val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("log1p MSE")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / "loss_curves.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved loss curve plot: {out_path}")


def plot_predictions(
    model_path: Path,
    dataset_path: Path,
    out_dir: Path,
    n_samples: int = 5,
    device: Optional[torch.device] = None,
):
    """Plot prediction vs truth for sample trajectories."""
    
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    # Load model
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    
    # Load dataset to infer dimensions
    ds = ODEDataset(dataset_path)
    y0_ex, u_ex, _, _ = ds[0]
    P = int(y0_ex.shape[0])
    U = int(u_ex.shape[1])
    
    # Build jump matrix
    control = ds.control_indices.tolist()
    obs = ds.obs_indices.tolist()
    jump = torch.zeros(U, P, dtype=torch.float32)
    obs_pos = {full_idx: p for p, full_idx in enumerate(obs)}
    for u, full_idx in enumerate(control):
        p = obs_pos.get(full_idx)
        if p is not None:
            jump[u, p] = 1.0
    
    # Recreate model
    cfg = ckpt.get("cfg", {})
    hidden = cfg.get("hidden", 128)
    num_layers = cfg.get("num_layers", 1)
    
    model = SimpleRNN(U, P=P, hidden=hidden, num_layers=num_layers, u_to_y_jump=jump)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Create dataloader for samples
    loader = DataLoader(
        ds,
        batch_size=min(n_samples, len(ds)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    
    y0_batch, u_batch, dt_batch, y_batch = next(iter(loader))
    y0_batch = y0_batch.to(device)
    u_batch = u_batch.to(device)
    dt_batch = dt_batch.to(device)
    y_batch = y_batch.to(device)
    
    with torch.no_grad():
        pred_batch, _ = model(y0_batch, u_batch, dt_batch, y_seq=None, teacher_forcing=False)
    
    # Plot each sample
    species_names = ["A", "D", "G", "J", "M"]
    
    for idx in range(min(n_samples, y0_batch.shape[0])):
        dt_np = dt_batch[idx].cpu().numpy()
        t = np.concatenate([[0.0], np.cumsum(dt_np)])
        y_true = y_batch[idx].cpu().numpy()
        y_pred = pred_batch[idx].cpu().numpy()
        
        fig, axes = plt.subplots(P, 1, figsize=(10, 2*P), sharex=True)
        if P == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            ax.plot(t[1:], y_true[:, i], label="true", linewidth=2, alpha=0.8)
            ax.plot(t[1:], y_pred[:, i], label="pred", linewidth=2, linestyle="--", alpha=0.8)
            ax.set_ylabel(species_names[i])
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.legend(loc="best")
        
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"Prediction vs Truth (Sample {idx})")
        fig.tight_layout()
        
        out_path = out_dir / f"pred_vs_true_sample{idx:03d}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
    
    print(f"Saved {n_samples} prediction plots to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot training diagnostics from saved results")
    parser.add_argument("--log-dir", type=str, required=True, help="Log directory containing loss_curves.npz")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model .pt file")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset .npz file")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of prediction plots to generate")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (defaults to log-dir)")
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    model_path = Path(args.model)
    dataset_path = Path(args.data)
    out_dir = Path(args.out_dir) if args.out_dir else log_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot loss curves
    loss_file = log_dir / "loss_curves.npz"
    if loss_file.exists():
        plot_loss_curves(loss_file, out_dir)
    else:
        print(f"Warning: {loss_file} not found, skipping loss curves")
    
    # Plot predictions
    if model_path.exists() and dataset_path.exists():
        plot_predictions(model_path, dataset_path, out_dir, n_samples=args.n_samples)
    else:
        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}")
        if not dataset_path.exists():
            print(f"Warning: Dataset not found at {dataset_path}")


if __name__ == "__main__":
    main()