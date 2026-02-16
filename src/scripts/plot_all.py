#!/usr/bin/env python3
"""
Unified plotting script - generates all training diagnostics in one command.
Automatically organizes plots into a clean directory structure.
"""
import argparse
from pathlib import Path
import sys

# Import plotting functions from other scripts
from src.scripts.plot_training_results import plot_loss_curves, plot_predictions
from src.scripts.plot_learned_params import plot_learned_parameters
from src.scripts.plot_epoch_evolution import plot_evolution


def plot_all_diagnostics(
    model_path: str,
    dataset_path: str,
    log_dir: str = None,
    n_samples: int = 5,
    sample_idx: int = 0,
    output_dir: str = None,
):
    """
    Generate all training diagnostic plots in an organized structure.
    
    Parameters
    ----------
    model_path : str
        Path to trained model (.pt file)
    dataset_path : str
        Path to dataset (.npz file)
    log_dir : str, optional
        Path to log directory. If None, infers from model filename.
    n_samples : int
        Number of prediction samples to plot
    sample_idx : int
        Sample index for time-varying parameter plots
    output_dir : str, optional
        Output directory for plots. If None, uses plots/{model_name}/
    """
    model_path = Path(model_path)
    dataset_path = Path(dataset_path)
    
    # Validate inputs
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Infer log directory if not provided
    if log_dir is None:
        # Extract timestamp from model filename
        # e.g., "N1000_T300_steps600_zeros_knoise0.0_ep200_b300_lr0.001_seed42_20260209_120348.pt"
        model_stem = model_path.stem
        parts = model_stem.split('_')
        
        # Find the pattern: ep{N}_b{N}_lr{N}_seed{N}_{timestamp}
        for i, part in enumerate(parts):
            if part.startswith('ep') and i + 4 < len(parts):
                # Construct log dir name
                log_name = '_'.join(parts[i:])
                log_dir = Path('logs') / log_name
                break
        
        if log_dir is None:
            raise ValueError(f"Could not infer log directory from model path: {model_path}")
    else:
        log_dir = Path(log_dir)
    
    if not log_dir.exists():
        print(f"Warning: Log directory not found: {log_dir}")
        print("Some plots may be skipped.")
    
    # Create organized output directory
    model_name = model_path.stem
    if output_dir is None:
        output_dir = Path('plots') / model_name
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Generating diagnostic plots for: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    # 1. Loss curves
    print("[1/4] Plotting loss curves...")
    loss_file = log_dir / "loss_curves.npz"
    if loss_file.exists():
        try:
            plot_loss_curves(loss_file, output_dir)
            print("      ✓ Loss curves saved")
        except Exception as e:
            print(f"      ✗ Failed: {e}")
    else:
        print(f"      ⊗ Skipped (loss file not found)")
    
    # 2. Prediction samples
    print(f"\n[2/4] Plotting {n_samples} prediction samples...")
    try:
        plot_predictions(model_path, dataset_path, output_dir, n_samples=n_samples)
        print(f"      ✓ {n_samples} prediction plots saved")
    except Exception as e:
        print(f"      ✗ Failed: {e}")
    
    # 3. Time-varying parameters (final model)
    print(f"\n[3/4] Plotting learned time-varying parameters (sample {sample_idx})...")
    try:
        theta_out = output_dir / f"theta_sample{sample_idx}.png"
        plot_learned_parameters(
            model_path=str(model_path),
            dataset_path=str(dataset_path),
            sample_idx=sample_idx,
            output_path=str(theta_out),
        )
        print(f"      ✓ Parameter plot saved")
    except Exception as e:
        print(f"      ✗ Failed: {e}")
    
    # 4. Epoch evolution (if checkpoints exist)
    print(f"\n[4/4] Plotting epoch evolution (sample {sample_idx})...")
    ckpt_dir = log_dir / "checkpoints"
    if ckpt_dir.exists() and any(ckpt_dir.glob("model_ep*.pt")):
        try:
            plot_evolution(
                log_dir=str(log_dir),
                dataset_path=str(dataset_path),
                sample_idx=sample_idx,
                out_dir=str(output_dir),
            )
            print(f"      ✓ Epoch evolution plots saved")
            print(f"      ✓ Time-varying parameter evolution saved to epoch_*/ subdirectories")
        except Exception as e:
            print(f"      ✗ Failed: {e}")
    else:
        print(f"      ⊗ Skipped (no checkpoints found)")
    
    print(f"\n{'='*70}")
    print(f"✓ All plots saved to: {output_dir}")
    print(f"{'='*70}\n")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate all training diagnostic plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using experiment name (recommended)
  python -m src.scripts.plot_all --exp-name "baseline_log10_loss"
  
  # Basic usage with model path (auto-detects log directory)
  python -m src.scripts.plot_all \\
      --model models/N1000_ep200_seed42_20260209_120348.pt \\
      --data datasets/N1000_T300_steps600_zeros_knoise0.0.npz
  
  # Specify log directory explicitly
  python -m src.scripts.plot_all \\
      --model models/my_model.pt \\
      --data datasets/my_data.npz \\
      --log-dir logs/ep200_b300_lr0.001_seed42_20260209_120348
  
  # Customize number of samples
  python -m src.scripts.plot_all \\
      --model models/my_model.pt \\
      --data datasets/my_data.npz \\
      --n-samples 10 \\
      --sample 5
        """
    )
    parser.add_argument("--exp-name", type=str, default=None,
                       help="Experiment name (auto-detects model/data/logs from experiments/{exp_name}/)")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to trained model (.pt file)")
    parser.add_argument("--data", type=str, default=None,
                       help="Path to dataset (.npz file)")
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Path to log directory (auto-detected if not provided)")
    parser.add_argument("--n-samples", type=int, default=5,
                       help="Number of prediction samples to plot (default: 5)")
    parser.add_argument("--sample", type=int, default=0,
                       help="Sample index for parameter plots (default: 0)")
    
    args = parser.parse_args()
    
    # Handle --exp-name: auto-detect paths from experiments/{exp_name}/
    if args.exp_name is not None:
        exp_dir = Path("experiments") / args.exp_name
        if not exp_dir.exists():
            print(f"✗ Error: Experiment not found: {exp_dir}", file=sys.stderr)
            sys.exit(1)
        
        # Auto-detect paths
        model_path = exp_dir / "model.pt"
        log_dir = exp_dir / "logs"
        output_dir = exp_dir / "plots"
        
        if not model_path.exists():
            print(f"✗ Error: Model not found in experiment: {model_path}", file=sys.stderr)
            sys.exit(1)
        
        # Try to find dataset from config.yaml
        config_path = exp_dir / "config.yaml"
        dataset_path = None
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            dataset_path = config.get('dataset', {}).get('path')
        
        if dataset_path is None or not Path(dataset_path).exists():
            # Fallback: user must provide --data
            if args.data is None:
                print(f"✗ Error: Could not auto-detect dataset path. Please provide --data argument.", file=sys.stderr)
                sys.exit(1)
            dataset_path = args.data
        
        print(f"Using experiment: {args.exp_name}")
        print(f"  Model:   {model_path}")
        print(f"  Dataset: {dataset_path}")
        print(f"  Logs:    {log_dir}")
        print(f"  Output:  {output_dir}")
        
        try:
            plot_all_diagnostics(
                model_path=str(model_path),
                dataset_path=str(dataset_path),
                log_dir=str(log_dir),
                n_samples=args.n_samples,
                sample_idx=args.sample,
                output_dir=str(output_dir),
            )
        except Exception as e:
            print(f"\n✗ Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Legacy mode: --model and --data required
    elif args.model is not None and args.data is not None:
        try:
            plot_all_diagnostics(
                model_path=args.model,
                dataset_path=args.data,
                log_dir=args.log_dir,
                n_samples=args.n_samples,
                sample_idx=args.sample,
            )
        except Exception as e:
            print(f"\n✗ Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        print("✗ Error: Either --exp-name or both --model and --data are required.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
