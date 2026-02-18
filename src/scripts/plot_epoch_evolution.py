"""
Plot how model predictions evolve over training epochs.
Loads periodic checkpoints and visualizes predictions on a fixed sample.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
# from src.models.simple_ode_rnn import SimpleRNN
from src.models.simple_ode_rnn_full_model import FullModelRNN as SimpleRNN
from src.data.ode_dataset import ODEDataset


def load_checkpoint(ckpt_path: Path, device: str, in_u: int, P: int, u_to_y_jump: torch.Tensor):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Create model with config from checkpoint
    cfg = ckpt.get("cfg", {})
    model = SimpleRNN(
        in_u=in_u,
        P=P,
        hidden=cfg.get("hidden", 128),
        num_layers=cfg.get("num_layers", 1),
        u_to_y_jump=u_to_y_jump,
    ).to(device)
    
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    epoch = ckpt.get("epoch", 0)
    return model, epoch


def predict_sample(model, y0, u_seq, dt_seq, device):
    """Run model inference on a single sample."""
    y0_t = torch.from_numpy(y0).float().unsqueeze(0).to(device)
    u_seq_t = torch.from_numpy(u_seq).float().unsqueeze(0).to(device)
    dt_seq_t = torch.from_numpy(dt_seq).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        y_pred, _ = model(y0_t, u_seq_t, dt_seq_t, y_seq=None, teacher_forcing=False)
    
    return y_pred.squeeze(0).cpu().numpy()


def plot_evolution(log_dir: str, dataset_path: str, sample_idx: int = 0, out_dir: str = None):
    """
    Plot how predictions evolve over training epochs.
    
    Args:
        log_dir: Path to training log directory containing checkpoints/
        dataset_path: Path to dataset .npz file
        sample_idx: Which sample to visualize (default: 0)
        out_dir: Where to save plot (default: log_dir)
    """
    log_dir = Path(log_dir)
    ckpt_dir = log_dir / "checkpoints"
    
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints found at {ckpt_dir}")
    
    # Load all checkpoints
    ckpt_paths = sorted(ckpt_dir.glob("model_ep*.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
    
    print(f"Found {len(ckpt_paths)} checkpoints")
    
    # Load dataset
    dataset = ODEDataset(dataset_path)
    y0, u_seq, dt_seq, y_true = dataset[sample_idx]
    
    # Infer dimensions
    P = y0.shape[0]  # number of observed species
    in_u = u_seq.shape[1]  # number of control inputs
    
    # Build u_to_y_jump matrix (same logic as in train.py)
    control = dataset.control_indices.tolist()
    obs = dataset.obs_indices.tolist()
    jump = torch.zeros(in_u, P, dtype=torch.float32)
    obs_pos = {full_idx: p for p, full_idx in enumerate(obs)}
    for u, full_idx in enumerate(control):
        p = obs_pos.get(full_idx)
        if p is not None:
            jump[u, p] = 1.0
    
    # Convert to numpy for prediction
    y0_np = y0.numpy()
    u_seq_np = u_seq.numpy()
    dt_seq_np = dt_seq.numpy()
    y_true_np = y_true.numpy()
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Reconstruct time axis
    t_obs = np.concatenate([[0], np.cumsum(dt_seq_np)])
    
    # Run predictions for each checkpoint
    epochs = []
    predictions = []
    
    for ckpt_path in ckpt_paths:
        model, epoch = load_checkpoint(ckpt_path, device, in_u, P, jump)
        y_pred = predict_sample(model, y0_np, u_seq_np, dt_seq_np, device)
        epochs.append(epoch)
        predictions.append(y_pred)
    
    # Create multi-panel plot
    n_species = y_true_np.shape[1]
    all_species_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]
    species_names = all_species_names[:n_species] if n_species <= 13 else [f"S{i}" for i in range(n_species)]
    
    fig, axes = plt.subplots(n_species, 1, figsize=(12, max(8, n_species * 0.8)), sharex=True)
    
    for i, ax in enumerate(axes):
        # Plot ground truth
        ax.plot(t_obs[1:], y_true_np[:, i], 'k-', linewidth=2, label='Truth', alpha=0.8)
        
        # Plot predictions with gradient color (early=light, late=dark)
        cmap = plt.cm.viridis
        for j, (epoch, y_pred) in enumerate(zip(epochs, predictions)):
            color = cmap(j / (len(epochs) - 1))
            alpha = 0.3 + 0.5 * (j / (len(epochs) - 1))  # Increase alpha over time
            label = f"Epoch {epoch}" if j % max(1, len(epochs) // 5) == 0 else None
            ax.plot(t_obs[1:], y_pred[:, i], color=color, alpha=alpha, linewidth=1, label=label)
        
        ax.set_ylabel(f"{species_names[i]} concentration", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel("Time", fontsize=12)
    fig.suptitle(f"Prediction Evolution Over Training (Sample {sample_idx})\n"
                 f"Epochs: {epochs[0]} → {epochs[-1]}", fontsize=14)
    
    plt.tight_layout()
    
    # Save plot
    if out_dir is None:
        out_dir = log_dir
    else:
        out_dir = Path(out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"epoch_evolution_sample{sample_idx}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot prediction evolution over training epochs")
    parser.add_argument("--log-dir", required=True, help="Path to training log directory")
    parser.add_argument("--data", required=True, help="Path to dataset .npz file")
    parser.add_argument("--sample", type=int, default=0, help="Sample index to visualize")
    parser.add_argument("--out-dir", default=None, help="Output directory for plot")
    
    args = parser.parse_args()
    plot_evolution(args.log_dir, args.data, args.sample, args.out_dir)


if __name__ == "__main__":
    main()
