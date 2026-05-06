"""
Replot / re-run R² for experiment runs that are missing plots or endpoint_r2.png.

Usage:
    python last-layer-ode/replot.py experiments/txtl_supervisor_combined_sweep
    python last-layer-ode/replot.py experiments/txtl_supervisor_combined_sweep --r2-only
    python last-layer-ode/replot.py experiments/txtl_supervisor_combined_sweep --plots-only
    python last-layer-ode/replot.py experiments/txtl_supervisor_combined_sweep --force
"""
from __future__ import annotations

import argparse
import sys
import traceback
import yaml
from pathlib import Path

# Ensure last-layer-ode is on sys.path for imports (same pattern as train_R.py)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def _needs_plots(exp_dir: Path, force: bool) -> bool:
    if force:
        return True
    plots_dir = exp_dir / "plots"
    if not plots_dir.exists():
        return True
    return not any(plots_dir.glob("*.png"))


def _needs_r2(exp_dir: Path, force: bool) -> bool:
    if force:
        return True
    return not (exp_dir / "endpoint_r2.png").exists()


def _config_wants_r2(exp_dir: Path) -> bool:
    cfg_path = exp_dir / "config.yaml"
    if not cfg_path.exists():
        return False
    cfg = yaml.safe_load(cfg_path.read_text())
    return bool(cfg.get("endpoint_r2", False))


def replot(
    sweep_dir: Path,
    *,
    do_plots: bool = True,
    do_r2: bool = True,
    force: bool = False,
    n_samples: int = 5,
    sample_idx: int = 0,
) -> None:
    exp_dirs = sorted(p for p in sweep_dir.iterdir() if p.is_dir() and (p / "model.pt").exists())

    if not exp_dirs:
        print(f"No completed runs found in {sweep_dir} (looked for model.pt).")
        return

    print(f"Found {len(exp_dirs)} completed run(s) in {sweep_dir}\n")

    for exp_dir in exp_dirs:
        name = exp_dir.name
        run_plots = do_plots and _needs_plots(exp_dir, force)
        run_r2 = do_r2 and (force or (_needs_r2(exp_dir, force) and _config_wants_r2(exp_dir)))

        if not run_plots and not run_r2:
            print(f"[skip]  {name}  (already complete)")
            continue

        actions = []
        if run_plots:
            actions.append("plots")
        if run_r2:
            actions.append("R²")
        print(f"\n[run]   {name}  ({', '.join(actions)})")

        if run_plots:
            try:
                from plot_diagnostics import plot_experiment, plot_epoch_prediction_overlays
                plot_experiment(exp_dir, n_samples=n_samples, sample_idx=sample_idx)
                plot_epoch_prediction_overlays(
                    exp_dir,
                    sample_idx=sample_idx,
                    epochs=None,
                    max_overlays=8,
                )
                print(f"  [plots] OK  →  {exp_dir / 'plots'}")
            except Exception:
                print(f"  [plots] FAILED:")
                traceback.print_exc()

        if run_r2:
            try:
                from metrics import endpoint_r2
                from plot_diagnostics import device_auto

                device = device_auto()
                result = endpoint_r2.collect_endpoints(
                    exp_dir, device, split="test", protein_sp="pm", mrna_sp="mm"
                )
                r2_protein = endpoint_r2.r2(result["true_protein_final"], result["pred_protein_final"])
                r2_mrna    = endpoint_r2.r2(result["true_mrna_max"],      result["pred_mrna_max"])
                print(f"  R²(protein final) = {r2_protein:.4f}")
                print(f"  R²(mRNA max)      = {r2_mrna:.4f}")

                out_path = exp_dir / "endpoint_r2.png"
                endpoint_r2.plot_endpoints(
                    [result], protein_sp="pm", mrna_sp="mm", split="test", out_path=out_path
                )
                endpoint_r2.save_r2_cache(exp_dir, result, r2_protein, r2_mrna)
                print(f"  [R²]   OK  →  {out_path}")
            except Exception:
                print(f"  [R²]   FAILED:")
                traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-generate missing plots / R² for a sweep folder.")
    parser.add_argument("sweep_dir", type=Path,
                        help="Sweep folder, e.g. experiments/txtl_supervisor_combined_sweep")
    parser.add_argument("--plots-only", action="store_true", help="Skip R², only redo plots")
    parser.add_argument("--r2-only",    action="store_true", help="Skip plots, only redo R²")
    parser.add_argument("--force",      action="store_true", help="Redo even if outputs already exist")
    parser.add_argument("--n-samples",  type=int, default=5)
    parser.add_argument("--sample-idx", type=int, default=0)
    args = parser.parse_args()

    replot(
        args.sweep_dir,
        do_plots=not args.r2_only,
        do_r2=not args.plots_only,
        force=args.force,
        n_samples=args.n_samples,
        sample_idx=args.sample_idx,
    )
