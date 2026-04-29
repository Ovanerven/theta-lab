"""Outlier analysis for endpoint R² (protein final and mRNA max).

Computes per-sample endpoint errors and lists the worst cases, plus optional
inspection of low-protein trajectories.

Usage:
  python last-layer-ode/metrics/endpoint_r2_outliers.py \
    experiments/txtl_fix_mse_minmax_seeds/20260428_190744_mse_minmax_seed1 \
    --split test --top-k 15 --pm-threshold 1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_diagnostics import rebuild_model_from_experiment, device_auto
from metrics import endpoint_r2


def load_split_indices(exp_dir: Path, split: str, n: int) -> list[int]:
    if split == "all":
        return list(range(n))
    split_path = exp_dir / "split.npz"
    if not split_path.exists():
        print(f"[warn] no split.npz in {exp_dir} — using full dataset")
        return list(range(n))
    idx = np.load(split_path)[f"{split}_idx"].tolist()
    if not idx:
        print(f"[warn] {split}_idx is empty — using full dataset")
        return list(range(n))
    return idx


def collect(exp_dir: Path, split: str, protein_sp: str, mrna_sp: str, device: torch.device):
    model, ds, obs_names, _ = rebuild_model_from_experiment(exp_dir, device)
    dt = torch.tensor(ds.dt.astype(np.float32)).to(device)
    obs_idx = torch.arange(len(obs_names), device=device)

    if protein_sp not in obs_names or mrna_sp not in obs_names:
        raise ValueError(
            f"species {protein_sp!r}/{mrna_sp!r} not in obs_names={obs_names}"
        )
    p_idx = obs_names.index(protein_sp)
    m_idx = obs_names.index(mrna_sp)

    indices = load_split_indices(exp_dir, split, len(ds))

    true_final, pred_final = [], []
    true_max, pred_max = [], []
    sample_idx = []

    model.eval()
    with torch.no_grad():
        for i in indices:
            y0, u_seq, y_seq = ds[i]
            pred, _, _ = model(
                y0.unsqueeze(0).to(device),
                u_seq.unsqueeze(0).to(device),
                dt.unsqueeze(0),
                obs_idx,
                y_seq=None,
                teacher_forcing=False,
            )
            y_np = y_seq.cpu().numpy()
            p_np = pred[0].cpu().numpy()

            sample_idx.append(i)
            true_final.append(y_np[-1, p_idx])
            pred_final.append(p_np[-1, p_idx])
            true_max.append(y_np[:, m_idx].max())
            pred_max.append(p_np[:, m_idx].max())

    return {
        "idx": np.array(sample_idx),
        "true_protein_final": np.array(true_final),
        "pred_protein_final": np.array(pred_final),
        "true_mrna_max": np.array(true_max),
        "pred_mrna_max": np.array(pred_max),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", help="Experiment run directory")
    parser.add_argument("--split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--protein", default="pm")
    parser.add_argument("--mrna", default="mm")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--pm-threshold", type=float, default=None,
                        help="Report samples with true protein final <= threshold")
    parser.add_argument("--out", type=str, default=None,
                        help="CSV path for per-sample endpoints and errors")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        raise SystemExit(f"Experiment directory not found: {exp_dir}")

    device = device_auto()
    print(f"Device: {device}   split: {args.split}")

    data = collect(exp_dir, args.split, args.protein, args.mrna, device)

    r2_p = endpoint_r2.r2(data["true_protein_final"], data["pred_protein_final"])
    r2_m = endpoint_r2.r2(data["true_mrna_max"], data["pred_mrna_max"])
    print(f"R²(protein final) = {r2_p:.4f}")
    print(f"R²(mRNA max)      = {r2_m:.4f}")

    err_p = data["true_protein_final"] - data["pred_protein_final"]
    err_m = data["true_mrna_max"] - data["pred_mrna_max"]
    sse = err_p ** 2 + err_m ** 2

    order = np.argsort(-sse)
    print("\nTop outliers by endpoint SSE:")
    print(f"{'idx':>6}  {'true_p':>10}  {'pred_p':>10}  {'true_m':>10}  {'pred_m':>10}  {'sse':>10}")
    for j in order[: args.top_k]:
        print(f"{int(data['idx'][j]):6d}  {data['true_protein_final'][j]:10.3f}  {data['pred_protein_final'][j]:10.3f}  {data['true_mrna_max'][j]:10.3f}  {data['pred_mrna_max'][j]:10.3f}  {sse[j]:10.3f}")

    if args.pm_threshold is not None:
        mask = data["true_protein_final"] <= args.pm_threshold
        idxs = np.where(mask)[0]
        print(f"\nTrue protein final <= {args.pm_threshold} (n={len(idxs)}):")
        print(f"{'idx':>6}  {'true_p':>10}  {'pred_p':>10}  {'true_m':>10}  {'pred_m':>10}  {'sse':>10}")
        for j in idxs[: args.top_k]:
            print(f"{int(data['idx'][j]):6d}  {data['true_protein_final'][j]:10.3f}  {data['pred_protein_final'][j]:10.3f}  {data['true_mrna_max'][j]:10.3f}  {data['pred_mrna_max'][j]:10.3f}  {sse[j]:10.3f}")

    out_path = Path(args.out) if args.out else exp_dir / "endpoint_outliers.csv"
    out = np.column_stack([
        data["idx"],
        data["true_protein_final"],
        data["pred_protein_final"],
        data["true_mrna_max"],
        data["pred_mrna_max"],
        err_p,
        err_m,
        sse,
    ])
    header = "idx,true_protein_final,pred_protein_final,true_mrna_max,pred_mrna_max,err_protein,err_mrna,sse"
    np.savetxt(out_path, out, delimiter=",", header=header, comments="")
    print(f"\nSaved per-sample endpoints -> {out_path}")


if __name__ == "__main__":
    main()
