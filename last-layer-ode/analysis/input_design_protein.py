"""OOD input-schedule diagnostic: use a trained surrogate to ask which reagent
feeding schedule maximizes predicted protein (pm), including out-of-distribution
dose magnitudes the wet-lab grid never tried.

This is a DIAGNOSTIC of what the encoder->theta->ODE surrogate has learned, not a
wet-lab recommendation: the model's OOD predictions are only trustworthy to the
extent the learned theta-map extrapolates sensibly. We therefore report (a) the
in-distribution dose-response shape per channel and (b) where the predicted
optimum lands relative to the training range, so monotonicity / saturation /
runaway can be judged by eye.

Usage:
    python last-layer-ode/analysis/input_design_protein.py \
        experiments_final/FINAL_coarse_transformer_inputs/20260610_095720_TI_decay_s0 \
        --n 16 --channels "DNA c,NTPs,AA,Mg-Glut,T7RNAP" --scales 0.5,1,2,4,8
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))          # last-layer-ode/
from plot_diagnostics import (                     # noqa: E402
    rebuild_model_from_experiment, device_auto, _test_subset, _filter_model_kwargs,
)
import yaml  # noqa: E402


def _protein_col(species_labels: list[str]) -> int:
    """Index into the model's observed-species output that is mature protein."""
    for i, s in enumerate(species_labels):
        if s.lower() in ("pm", "p_fluor", "protein", "p"):
            return i
    return len(species_labels) - 1   # fall back to the last observed species


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--n", type=int, default=16, help="# test trajectories to average over")
    ap.add_argument("--channels", type=str, default="DNA c,NTPs,AA,Mg-Glut,T7RNAP")
    ap.add_argument("--scales", type=str, default="0.25,0.5,1,2,4,8")
    args = ap.parse_args()

    device = device_auto()
    cfg = yaml.safe_load((args.run_dir / "config.yaml").read_text())
    model, ds, obs_names, _, lift_info = rebuild_model_from_experiment(args.run_dir, device)
    model.eval()

    # control-channel names live on the dataset npz
    dpath = cfg["dataset_path"]
    dnpz = np.load(dpath, allow_pickle=True)
    ctrl_names = [str(x) for x in dnpz["control_names"]]
    name_to_col = {n: i for i, n in enumerate(ctrl_names)}

    if lift_info:
        scaffold_obs = list(lift_info["scaffold_obs_idx"])
        species_labels = [obs_names[i] if i < len(obs_names) else f"y{i}" for i in scaffold_obs]
    else:
        scaffold_obs = list(range(len(obs_names)))
        species_labels = list(obs_names)
    p_idx = _protein_col(species_labels)
    print(f"observed species = {species_labels}  | protein col = {p_idx} ({species_labels[p_idx]})")
    print(f"control channels = {ctrl_names}")

    # gather per-trajectory tensors (trajectories have unequal length → no stacking)
    test_subset = _test_subset(ds, args.run_dir)
    n = min(args.n, len(test_subset))
    items = [(test_subset[i][0].to(device),      # y0   (dataset state dim)
              test_subset[i][1].to(device),       # u_seq  (T_i, C)
              test_subset[i][2].to(device),       # y_seq  (T_i, obs) — for lifting y0 only
              test_subset[i][3].to(device))        # dt     (T_i,)
             for i in range(n)]
    if lift_info:
        from plot_diagnostics import _maybe_lift
        obs_idx_t = torch.tensor(scaffold_obs, device=device, dtype=torch.long)
    else:
        _maybe_lift = None
        obs_idx_t = torch.arange(len(species_labels), device=device)
    base_kwargs = {
        "y_seq": None, "teacher_forcing": False,
        "u_transform": str(cfg.get("u_transform", "none")),
        "y_transform": str(cfg.get("y_transform", "none")),
    }
    kw = _filter_model_kwargs(model, base_kwargs)

    def predict_protein(scale_col: "int | None" = None, scale: float = 1.0) -> float:
        """Mean over trajectories of predicted FINAL protein, optionally scaling
        one reagent channel's whole feeding profile by `scale`."""
        vals = []
        with torch.no_grad():
            for y0_i, u_i, y_i, dt_i in items:
                u = u_i.clone()
                if scale_col is not None:
                    u[:, scale_col] = u[:, scale_col] * scale
                y0_b = y0_i.unsqueeze(0)
                if _maybe_lift is not None:
                    y0_b, _ = _maybe_lift(y0_b, y_i.unsqueeze(0), lift_info)
                pred, _, _ = model(y0_b, u.unsqueeze(0),
                                   dt_i.unsqueeze(0), obs_idx_t, **kw)
                pf = pred[0, -1, p_idx]
                vals.append(float(torch.nan_to_num(pf)))
        return float(np.mean(vals))

    base = predict_protein()
    print(f"\nbaseline mean final protein (×1.0 schedule) = {base:.4f}\n")

    scales = [float(s) for s in args.scales.split(",")]
    channels = [c.strip() for c in args.channels.split(",")]
    print(f"{'channel':14s} | " + " ".join(f"x{s:<6g}" for s in scales) + " | best")
    print("-" * (16 + 8 * len(scales) + 8))
    results = {}
    for ch in channels:
        if ch not in name_to_col:
            print(f"{ch:14s} | (not a control channel — skipped)")
            continue
        col = name_to_col[ch]
        row = [predict_protein(col, s) for s in scales]
        results[ch] = row
        best_s = scales[int(np.argmax(row))]
        ood = " (OOD)" if best_s > 1.0 else ""
        cells = " ".join(f"{v:7.4f}" for v in row)
        print(f"{ch:14s} | {cells} | x{best_s:g}{ood}")

    # crude greedy multi-channel schedule: apply each channel's best single-axis scale
    best_scale = {name_to_col[ch]: scales[int(np.argmax(row))] for ch, row in results.items()}
    combo_vals = []
    with torch.no_grad():
        for y0_i, u_i, y_i, dt_i in items:
            u = u_i.clone()
            for col, s in best_scale.items():
                u[:, col] = u[:, col] * s
            y0_b = y0_i.unsqueeze(0)
            if _maybe_lift is not None:
                y0_b, _ = _maybe_lift(y0_b, y_i.unsqueeze(0), lift_info)
            pred, _, _ = model(y0_b, u.unsqueeze(0),
                               dt_i.unsqueeze(0), obs_idx_t, **kw)
            combo_vals.append(float(torch.nan_to_num(pred[0, -1, p_idx])))
    combo = float(np.mean(combo_vals))
    print(f"\ngreedy combined (each channel at its best scale) = {combo:.4f}  "
          f"({100*(combo-base)/max(abs(base),1e-6):+.0f}% vs baseline)")
    print("NOTE: greedy combo is almost certainly OOD on every axis at once — treat as a")
    print("      probe of the surrogate's belief, not a feasible protocol.")


if __name__ == "__main__":
    main()
