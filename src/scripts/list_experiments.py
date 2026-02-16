#!/usr/bin/env python3
"""
List and compare all experiments in the experiments/ directory.
Shows training config, final losses, and checkpoint information.
"""
import argparse
from pathlib import Path
import yaml
import numpy as np
from typing import Dict, List, Any, Optional


def load_experiment_info(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """Load experiment information from config.yaml and loss_curves.npz"""
    config_path = exp_dir / "config.yaml"
    loss_path = exp_dir / "logs" / "loss_curves.npz"
    
    info = {
        "name": exp_dir.name,
        "path": str(exp_dir),
        "config": None,
        "final_train_loss": None,
        "final_val_loss": None,
        "best_val_loss": None,
        "epochs_trained": None,
        "num_checkpoints": 0,
        "has_model": (exp_dir / "model.pt").exists(),
        "has_plots": (exp_dir / "plots").exists(),
    }
    
    # Load config
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                info["config"] = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config for {exp_dir.name}: {e}")
    
    # Load loss curves
    if loss_path.exists():
        try:
            data = np.load(loss_path, allow_pickle=True)
            train_losses = data.get('train_losses', None)
            val_losses = data.get('val_losses', None)
            
            if train_losses is not None and len(train_losses) > 0:
                info["final_train_loss"] = float(train_losses[-1])
                info["epochs_trained"] = len(train_losses)
            
            if val_losses is not None and len(val_losses) > 0:
                info["final_val_loss"] = float(val_losses[-1])
                info["best_val_loss"] = float(np.min(val_losses))
        except Exception as e:
            print(f"Warning: Could not load losses for {exp_dir.name}: {e}")
    
    # Count checkpoints
    ckpt_dir = exp_dir / "logs" / "checkpoints"
    if ckpt_dir.exists():
        info["num_checkpoints"] = len(list(ckpt_dir.glob("*.pt")))
    
    return info


def format_loss(loss: Optional[float]) -> str:
    """Format loss value for display"""
    if loss is None:
        return "N/A"
    return f"{loss:.6f}"


def format_number(num: Optional[int]) -> str:
    """Format number for display"""
    if num is None:
        return "N/A"
    return str(num)


def print_experiment_summary(info: Dict[str, Any], verbose: bool = False):
    """Print formatted experiment information"""
    print(f"\n{'='*80}")
    print(f"Experiment: {info['name']}")
    print(f"{'='*80}")
    
    # Training progress
    print(f"\n📊 Training Progress:")
    print(f"  Epochs Trained:      {format_number(info['epochs_trained'])}")
    print(f"  Final Train Loss:    {format_loss(info['final_train_loss'])}")
    print(f"  Final Val Loss:      {format_loss(info['final_val_loss'])}")
    print(f"  Best Val Loss:       {format_loss(info['best_val_loss'])}")
    
    # Artifacts
    print(f"\n📁 Artifacts:")
    print(f"  Model Saved:         {'✓' if info['has_model'] else '✗'}")
    print(f"  Plots Generated:     {'✓' if info['has_plots'] else '✗'}")
    print(f"  Checkpoints:         {info['num_checkpoints']}")
    
    # Configuration (if verbose)
    if verbose and info['config'] is not None:
        print(f"\n⚙️  Configuration:")
        
        # Training config
        training = info['config'].get('training', {})
        print(f"  Training:")
        print(f"    Batch Size:        {training.get('batch_size', 'N/A')}")
        print(f"    Learning Rate:     {training.get('learning_rate', 'N/A')}")
        print(f"    Weight Decay:      {training.get('weight_decay', 'N/A')}")
        print(f"    Grad Clip:         {training.get('grad_clip', 'N/A')}")
        print(f"    Seed:              {training.get('seed', 'N/A')}")
        
        # Model config
        model = info['config'].get('model', {})
        print(f"  Model:")
        print(f"    Hidden Size:       {model.get('hidden', 'N/A')}")
        print(f"    Num Layers:        {model.get('num_layers', 'N/A')}")
        
        # Dataset
        dataset = info['config'].get('dataset', {})
        dataset_path = dataset.get('path', 'N/A')
        if dataset_path != 'N/A':
            dataset_path = Path(dataset_path).name  # show filename only
        print(f"  Dataset:             {dataset_path}")
    
    print(f"\n📂 Location: {info['path']}")


def print_comparison_table(experiments: List[Dict[str, Any]]):
    """Print comparison table of all experiments"""
    if not experiments:
        print("No experiments found.")
        return
    
    print(f"\n{'='*140}")
    print(f"Experiment Comparison")
    print(f"{'='*140}")
    
    # Header
    header = f"{'Name':<30} {'Epochs':>8} {'Train Loss':>12} {'Val Loss':>12} {'Best Val':>12} {'Ckpts':>6} {'Model':>6} {'Plots':>6}"
    print(header)
    print('-' * 140)
    
    # Sort by best validation loss (if available)
    experiments_sorted = sorted(
        experiments,
        key=lambda x: x['best_val_loss'] if x['best_val_loss'] is not None else float('inf')
    )
    
    # Rows
    for info in experiments_sorted:
        name = info['name'][:28] + '..' if len(info['name']) > 30 else info['name']
        row = (
            f"{name:<30} "
            f"{format_number(info['epochs_trained']):>8} "
            f"{format_loss(info['final_train_loss']):>12} "
            f"{format_loss(info['final_val_loss']):>12} "
            f"{format_loss(info['best_val_loss']):>12} "
            f"{info['num_checkpoints']:>6} "
            f"{'✓':>6}" if info['has_model'] else f"{'✗':>6} "
            f"{'✓':>6}" if info['has_plots'] else f"{'✗':>6}"
        )
        print(row)
    
    print(f"{'='*140}\n")


def main():
    parser = argparse.ArgumentParser(
        description="List and compare experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments (summary table)
  python -m src.scripts.list_experiments
  
  # Show detailed info for specific experiment
  python -m src.scripts.list_experiments --exp-name "baseline_log10_loss" --verbose
  
  # Compare specific experiments
  python -m src.scripts.list_experiments --exp-name "exp1" "exp2" "exp3"
        """
    )
    parser.add_argument("--exp-name", type=str, nargs='+', default=None,
                       help="Specific experiment name(s) to show (shows all if not provided)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed configuration for each experiment")
    parser.add_argument("--sort-by", type=str, default="best_val",
                       choices=["name", "train_loss", "val_loss", "best_val", "epochs"],
                       help="Sort experiments by this metric (default: best_val)")
    
    args = parser.parse_args()
    
    # Find experiments directory
    exp_root = Path("experiments")
    if not exp_root.exists():
        print(f"No experiments directory found: {exp_root}")
        print("Create experiments using: python -m src.train.train --exp-name \"my_experiment\"")
        return
    
    # Gather experiment directories
    if args.exp_name is not None:
        # Specific experiments
        exp_dirs = [exp_root / name for name in args.exp_name]
        missing = [d for d in exp_dirs if not d.exists()]
        if missing:
            print(f"Error: Experiments not found: {[d.name for d in missing]}")
            return
    else:
        # All experiments
        exp_dirs = sorted([d for d in exp_root.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    if not exp_dirs:
        print("No experiments found.")
        print(f"Create experiments using: python -m src.train.train --exp-name \"my_experiment\"")
        return
    
    # Load experiment info
    experiments = []
    for exp_dir in exp_dirs:
        info = load_experiment_info(exp_dir)
        if info is not None:
            experiments.append(info)
    
    # Display results
    if len(experiments) == 1 or args.verbose:
        # Detailed view
        for info in experiments:
            print_experiment_summary(info, verbose=True)
    else:
        # Comparison table
        print_comparison_table(experiments)
        
        if experiments:
            print(f"💡 Tip: Use --exp-name <name> --verbose for detailed info")
            print(f"💡 Tip: Use --sort-by to change sorting (current: {args.sort_by})")


if __name__ == "__main__":
    main()
