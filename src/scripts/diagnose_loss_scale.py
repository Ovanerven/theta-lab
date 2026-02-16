#!/usr/bin/env python3
"""
Diagnose scale sensitivity in loss function.
Check if the model fits high-concentration species better than low-concentration ones.
"""
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.simple_ode_rnn import SimpleRNN
from src.data.ode_dataset import ODEDataset


def diagnose_loss_scale(model_path: str, dataset_path: str, n_samples: int = 100):
    """
    Analyze per-species errors to detect scale sensitivity.
    
    If high-concentration species have lower relative error than 
    low-concentration species, we have a scale problem.
    """
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # Load dataset
    ds = ODEDataset(dataset_path)
    n_samples = min(n_samples, len(ds))
    
    y0_ex, u_ex, _, _ = ds[0]
    P = y0_ex.shape[0]
    U = u_ex.shape[1]
    
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
    model = SimpleRNN(U, P=P, hidden=128, num_layers=1, u_to_y_jump=jump)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Collect predictions and ground truth
    all_true = []
    all_pred = []
    
    print(f"Analyzing {n_samples} samples...")
    
    with torch.no_grad():
        for idx in range(n_samples):
            y0, u_seq, dt_seq, y_seq = ds[idx]
            
            y0_batch = y0.unsqueeze(0)
            u_batch = u_seq.unsqueeze(0)
            dt_batch = dt_seq.unsqueeze(0)
            
            pred, _ = model(y0_batch, u_batch, dt_batch, y_seq=None, teacher_forcing=False)
            
            all_true.append(y_seq.numpy())  # (K, P)
            all_pred.append(pred.squeeze(0).numpy())  # (K, P)
    
    # Stack into (N, K, P)
    true_arr = np.array(all_true)
    pred_arr = np.array(all_pred)
    
    # Flatten to (N*K, P)
    true_flat = true_arr.reshape(-1, P)
    pred_flat = pred_arr.reshape(-1, P)
    
    # Compute per-species statistics
    species_names = ["A", "D", "G", "J", "M"]
    
    print("\n" + "="*70)
    print("Per-Species Error Analysis")
    print("="*70)
    
    results = []
    for i, name in enumerate(species_names):
        true_i = true_flat[:, i]
        pred_i = pred_flat[:, i]
        
        # Absolute error
        abs_error = np.abs(pred_i - true_i)
        mae = np.mean(abs_error)
        
        # Relative error (avoid division by zero)
        rel_error = np.abs((pred_i - true_i) / (true_i + 1e-8))
        mape = np.mean(rel_error) * 100
        
        # Mean concentration
        mean_conc = np.mean(true_i)
        
        # MSE contribution (what the model actually optimizes)
        mse_contrib = np.mean((pred_i - true_i)**2)
        
        results.append({
            'name': name,
            'mean_conc': mean_conc,
            'mae': mae,
            'mape': mape,
            'mse_contrib': mse_contrib,
        })
        
        print(f"\nSpecies {name}:")
        print(f"  Mean concentration: {mean_conc:.2e}")
        print(f"  MAE (absolute):     {mae:.2e}")
        print(f"  MAPE (relative):    {mape:.2f}%")
        print(f"  MSE contribution:   {mse_contrib:.2e}")
    
    print("\n" + "="*70)
    print("Diagnosis:")
    print("="*70)
    
    # Check if MSE contributions are proportional to concentration scales
    concs = [r['mean_conc'] for r in results]
    mse_contribs = [r['mse_contrib'] for r in results]
    mapes = [r['mape'] for r in results]
    
    # Correlation between concentration and MSE contribution
    conc_mse_corr = np.corrcoef(concs, mse_contribs)[0, 1]
    
    # Check if low-concentration species have worse relative error
    sorted_by_conc = sorted(results, key=lambda x: x['mean_conc'])
    low_conc_mape = np.mean([r['mape'] for r in sorted_by_conc[:2]])
    high_conc_mape = np.mean([r['mape'] for r in sorted_by_conc[-2:]])
    
    print(f"\n1. MSE contribution vs concentration correlation: {conc_mse_corr:.3f}")
    if conc_mse_corr > 0.7:
        print("   ⚠️  HIGH CORRELATION: MSE dominated by high-concentration species!")
    else:
        print("   ✓  Low correlation: MSE fairly balanced across species")
    
    print(f"\n2. Relative error comparison:")
    print(f"   Low-concentration species MAPE:  {low_conc_mape:.2f}%")
    print(f"   High-concentration species MAPE: {high_conc_mape:.2f}%")
    if low_conc_mape > 1.5 * high_conc_mape:
        print("   ⚠️  Low-concentration species have MUCH worse relative error!")
    else:
        print("   ✓  Relative errors are balanced")
    
    # Plot diagnostics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Mean concentration
    axes[0].bar(species_names, concs, color='steelblue')
    axes[0].set_ylabel('Mean Concentration', fontsize=11)
    axes[0].set_title('Concentration Scale', fontsize=12)
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: MSE contribution (what model optimizes)
    axes[1].bar(species_names, mse_contribs, color='coral')
    axes[1].set_ylabel('MSE Contribution', fontsize=11)
    axes[1].set_title('Loss Contribution (Current)', fontsize=12)
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Relative error (what we care about)
    axes[2].bar(species_names, mapes, color='seagreen')
    axes[2].set_ylabel('Mean Absolute % Error', fontsize=11)
    axes[2].set_title('Relative Prediction Error', fontsize=12)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    out_dir = Path('plots') / 'diagnostics'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'loss_scale_diagnosis.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Diagnostic plot saved to: {out_path}")
    
    plt.close()
    
    print("\n" + "="*70)
    print("Recommendations:")
    print("="*70)
    if conc_mse_corr > 0.7 or low_conc_mape > 1.5 * high_conc_mape:
        print("⚠️  SCALE SENSITIVITY DETECTED!")
        print("\nSuggested fixes:")
        print("1. Log-transform the data before computing loss")
        print("   loss = MSE(log(pred), log(true))")
        print("\n2. Log-transform during ODE integration")
        print("   Solve dx/dt in log-space: d(log x)/dt = (1/x) * dx/dt")
        print("\n3. Use relative error loss")
        print("   loss = mean(|pred - true| / true)")
    else:
        print("✓ No major scale sensitivity detected.")
        print("Current loss function appears adequate.")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Diagnose loss scale sensitivity")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples to analyze")
    
    args = parser.parse_args()
    diagnose_loss_scale(args.model, args.data, args.n_samples)


if __name__ == "__main__":
    main()
