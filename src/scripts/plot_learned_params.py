#!/usr/bin/env python3
"""
Plot learned time-varying ODE parameters (theta) from a trained model.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# from src.models.simple_ode_rnn import SimpleRNN
from src.models.simple_ode_rnn_full_model import FullModelRNN as SimpleRNN
from src.data.ode_dataset import ODEDataset


def plot_learned_parameters(
    model_path: str,
    dataset_path: str,
    sample_idx: int = 0,
    output_path: str = None,
):
    """
    Load trained model and dataset, run inference on a sample,
    and plot the predicted time-varying parameters theta(t).
    
    Parameters
    ----------
    model_path : str
        Path to saved model checkpoint (.pt file)
    dataset_path : str
        Path to dataset (.npz file)
    sample_idx : int
        Which sample from the dataset to visualize
    output_path : str, optional
        Where to save the plot. If None, uses model directory.
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # Load dataset
    ds = ODEDataset(dataset_path)
    
    if sample_idx >= len(ds):
        raise ValueError(f"sample_idx {sample_idx} out of range (dataset has {len(ds)} samples)")
    
    y0, u_seq, dt_seq, y_seq = ds[sample_idx]
    P = y0.shape[0]
    K, U = u_seq.shape
    
    # Build jump matrix
    control = ds.control_indices.tolist()
    obs = ds.obs_indices.tolist()
    jump = torch.zeros(U, P, dtype=torch.float32)
    obs_pos = {full_idx: p for p, full_idx in enumerate(obs)}
    for u, full_idx in enumerate(control):
        p = obs_pos.get(full_idx)
        if p is not None:
            jump[u, p] = 1.0
    
    # Reconstruct model
    model = SimpleRNN(U, P=P, hidden=128, num_layers=1, u_to_y_jump=jump)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Run inference (batch size 1)
    with torch.no_grad():
        y0_batch = y0.unsqueeze(0)      # (1, P) initial observed state
        u_batch = u_seq.unsqueeze(0)    # (1, K, U) control inputs
        dt_batch = dt_seq.unsqueeze(0)  # (1, K) time intervals
        
        pred, theta = model(y0_batch, u_batch, dt_batch, y_seq=None, teacher_forcing=False)
    
    # Extract results
    theta_np = theta[0].cpu().numpy()  # (K, n_params)
    dt_np = dt_seq.cpu().numpy()
    t = np.concatenate([[0.0], np.cumsum(dt_np)])  # (K+1,)
    t_theta = t[1:]  # theta is defined at t1..tK
    
    # Parameter names - handle both 8 (reduced) and 19 (full model)
    n_params = theta_np.shape[1]
    if n_params == 8:
        param_names = ['kf1', 'kf2', 'kf3', 'kf4', 'kr1', 'kr2', 'kr3', 'kr4']
        nrows, ncols = 4, 2
    elif n_params == 19:
        param_names = ['kf1', 'kf2', 'kf3', 'kf4', 'kf5', 'kf6', 'kf7', 'kf8', 'kf9', 'kf10', 'kf11', 'kf12',
                      'kr1', 'kr3', 'kr5', 'kr7', 'kr9', 'kr11', 'kr12']
        nrows, ncols = 7, 3  # 21 subplots (19 used)
    else:
        param_names = [f'θ{i+1}' for i in range(n_params)]
        ncols = min(3, n_params)
        nrows = (n_params + ncols - 1) // ncols
    
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 2*nrows), sharex=True)
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes[:n_params], param_names)):
        ax.plot(t_theta, theta_np[:, i], linewidth=1.5, color=f'C{i % 10}')
        ax.set_ylabel(name, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    # Hide extra subplots if any
    for i in range(n_params, len(axes)):
        axes[i].axis('off')
    
    # Set xlabel on bottom row
    for i in range(nrows * ncols - ncols, nrows * ncols):
        if i < len(axes):
            axes[i].set_xlabel('Time', fontsize=12)
    
    model_name = Path(model_path).stem
    fig.suptitle(f'Learned Time-Varying Parameters θ(t) - {model_name} - Sample {sample_idx}', fontsize=14, y=0.995)
    fig.tight_layout()
    
    # Save
    if output_path is None:
        model_dir = Path(model_path).parent
        output_path = model_dir / f"theta_{model_name}_sample{sample_idx}.png"
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved parameter plot to {output_path}")
    plt.close(fig)
    
    return theta_np, t_theta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot learned time-varying parameters")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset (.npz)")
    parser.add_argument("--sample", type=int, default=0, help="Sample index to visualize")
    parser.add_argument("--output", type=str, default=None, help="Output path for plot")
    
    args = parser.parse_args()
    
    plot_learned_parameters(
        model_path=args.model,
        dataset_path=args.data,
        sample_idx=args.sample,
        output_path=args.output,
    )
