"""Experimentally-relevant endpoint R^2: protein final yield and mRNA max.

For each run in an experiment folder, rolls out the model on the test split
(open-loop) and computes two endpoints per trajectory:

  * protein (default: species "pm"):  final value y[-1]
  * mRNA    (default: species "mm"):  max value over the trajectory

Then computes R^2 between predicted and true endpoints across all samples,
per run, and draws a 2-panel scatter (protein final vs mRNA max).

Usage:
    python last-layer-ode/metrics/endpoint_r2.py experiments/some_folder
    python last-layer-ode/metrics/endpoint_r2.py experiments/foo --split train
    python last-layer-ode/metrics/endpoint_r2.py experiments/foo \\
        --protein pm --mrna mm --out results/foo_endpoints
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_diagnostics import rebuild_model_from_experiment, device_auto, load_yaml, _filter_model_kwargs


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _split_subset(ds, exp_dir: Path, split: str):
    """Return the requested split subset. split ∈ {test, train, all}."""
    if split == "all":
        return ds
    split_path = exp_dir / "split.npz"
    if not split_path.exists():
        print(f"  [warn] no split.npz in {exp_dir.name} — using full dataset")
        return ds
    idx = np.load(split_path)[f"{split}_idx"].tolist()
    if not idx:
        print(f"  [warn] {split}_idx is empty in {exp_dir.name} — using full dataset")
        return ds
    return torch.utils.data.Subset(ds, idx)


def _find_runs(root: Path) -> list[Path]:
    if (root / "config.yaml").exists() and (root / "model.pt").exists():
        return [root]
    return sorted(
        d for d in root.iterdir()
        if d.is_dir()
        and (d / "config.yaml").exists()
        and (d / "model.pt").exists()
        and (d / "split.npz").exists()
    )


def collect_endpoints(exp_dir: Path, device: torch.device, split: str,
                      protein_sp: str, mrna_sp: str) -> dict:
    model, ds, obs_names, _, lift_info = rebuild_model_from_experiment(exp_dir, device)
    subset = _split_subset(ds, exp_dir, split)
    # dt is now read per-sample from the dataset's __getitem__ (item[3]); the
    # shared ds.dt only matters as a fallback when no per-sample grid exists.
    if lift_info:
        # Scaffold layout: obs_state_idx[0]=mm slot, obs_state_idx[1]=pm slot
        # (mirrors cfg.obs_idx convention: mRNA first, protein second).
        obs_idx = torch.tensor(lift_info["scaffold_obs_idx"], device=device, dtype=torch.long)
    else:
        obs_idx = torch.arange(len(obs_names), device=device)

    # Forward-time feature transforms (u_transform/y_transform) are kwargs on
    # ode_rnn.forward — they MUST be passed at eval time too, or the trained
    # model is fed raw (un-sqrt'd) features and produces garbage. bob_gru_verbatim
    # bakes these transforms into its forward and ignores the kwargs, so the
    # _filter_model_kwargs strip is safe for it.
    cfg_local = load_yaml(exp_dir / "config.yaml")
    base_kwargs = {
        "y_seq": None,
        "teacher_forcing": False,
        "u_transform": str(cfg_local.get("u_transform", "none")),
        "y_transform": str(cfg_local.get("y_transform", "none")),
    }

    if lift_info:
        # Partial-obs scaffold: pred is in scaffold layout. By convention
        # cfg.obs_idx = [mm_idx, pm_idx], so position 0 = mRNA, position 1 = protein
        # in scaffold_obs_idx as well. The dataset-side y_seq is read at the
        # source obs positions (dataset_obs_idx) for ground truth.
        m_idx = int(lift_info["scaffold_obs_idx"][0])
        p_idx = int(lift_info["scaffold_obs_idx"][1])
        m_idx_data = int(lift_info["dataset_obs_idx"][0])
        p_idx_data = int(lift_info["dataset_obs_idx"][1])
    else:
        if protein_sp not in obs_names or mrna_sp not in obs_names:
            raise ValueError(
                f"{exp_dir.name}: species {protein_sp!r}/{mrna_sp!r} not in obs_names={obs_names}"
            )
        p_idx = obs_names.index(protein_sp)
        m_idx = obs_names.index(mrna_sp)
        m_idx_data = m_idx
        p_idx_data = p_idx

    true_final, pred_final = [], []
    true_max,   pred_max   = [], []
    sources = []
    n_skipped_synth = 0

    # When the dataset carries a z_expr label (combined real + synthetic no-go
    # build), evaluate the endpoint metric only on the real experiments.
    # Synthetic no-go samples are flat-baseline by construction; including them
    # in R² collapses the metric onto "predict the baseline" and obscures
    # real-data fit quality. The subset itself still iterates everything; we
    # just skip the synth rows here.
    raw_ds = subset.dataset if isinstance(subset, torch.utils.data.Subset) else subset
    z_expr_arr  = getattr(raw_ds, "z_expr",  None)
    source_arr  = getattr(raw_ds, "source",  None)  # 0=old, 1=new real data
    subset_indices = (list(subset.indices) if isinstance(subset, torch.utils.data.Subset)
                      else list(range(len(raw_ds))))
    # Variable-length datasets pad y_seq to K with zeros past lengths[i]. Reading
    # y[-1] for "protein final" then yields the padded zero instead of the actual
    # experiment endpoint — collapses true_protein_final to all zeros on datasets
    # like txtl_native_new_only (lengths ≈ 270, K = 1497) and makes R² meaningless.
    lengths_arr = getattr(raw_ds, "lengths", None)

    model.eval()
    with torch.no_grad():
        for i in range(len(subset)):
            global_i = subset_indices[i]
            if z_expr_arr is not None and int(z_expr_arr[global_i]) == 0:
                n_skipped_synth += 1
                continue
            item = subset[i]
            y0, u_seq, y_seq, dt_i = item[0], item[1], item[2], item[3]
            y0_b = y0.unsqueeze(0).to(device)
            u_b = u_seq.unsqueeze(0).to(device)
            y_seq_b = y_seq.unsqueeze(0).to(device)
            dt_b = dt_i.unsqueeze(0).to(device)
            # Lift into scaffold layout if needed.
            if lift_info:
                from plot_diagnostics import _maybe_lift
                y0_b, _y_lift = _maybe_lift(y0_b, y_seq_b, lift_info)
            pred, _, _ = model(
                y0_b,
                u_b,
                dt_b,
                obs_idx,
                **_filter_model_kwargs(model, base_kwargs),
            )
            # Ground truth comes from the dataset (always); pred is in scaffold layout.
            y_np = y_seq.cpu().numpy()
            p_np = pred[0].cpu().numpy()

            K_full = y_np.shape[0]
            L_i = int(lengths_arr[global_i]) if lengths_arr is not None else K_full
            L_i = max(1, min(L_i, K_full, p_np.shape[0]))
            true_final.append(y_np[L_i - 1, p_idx_data])
            pred_final.append(p_np[L_i - 1, p_idx])
            true_max.append(y_np[:L_i, m_idx_data].max())
            pred_max.append(p_np[:L_i, m_idx].max())
            src = int(source_arr[global_i]) if source_arr is not None else -1
            sources.append(src)

    if n_skipped_synth:
        print(f"  endpoint_r2: skipped {n_skipped_synth} synthetic no-go samples; "
              f"R² computed on {len(true_final)} real experiments.")
    return {
        "run": exp_dir.name,
        "n": len(true_final),
        "n_skipped_synth": n_skipped_synth,
        "true_protein_final": np.array(true_final),
        "pred_protein_final": np.array(pred_final),
        "true_mrna_max":      np.array(true_max),
        "pred_mrna_max":      np.array(pred_max),
        "sources":            np.array(sources, dtype=np.int32),
    }


def plot_endpoints(results: list[dict], protein_sp: str, mrna_sp: str,
                   split: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    panels = [
        (axes[0], "true_protein_final", "pred_protein_final",
         f"Protein final yield ({protein_sp})"),
        (axes[1], "true_mrna_max",      "pred_mrna_max",
         f"mRNA max ({mrna_sp})"),
    ]

    for ax, tkey, pkey, title in panels:
        all_vals = []
        for i, res in enumerate(results):
            c = colors[i % len(colors)]
            r2_val = r2(res[tkey], res[pkey])
            ax.scatter(res[tkey], res[pkey], s=28, alpha=0.75, color=c,
                       edgecolors="white", linewidths=0.5,
                       label=f"{res['run']}  R²={r2_val:.3f}  (n={res['n']})")
            all_vals.extend(res[tkey]); all_vals.extend(res[pkey])

        if all_vals:
            lo, hi = float(min(all_vals)), float(max(all_vals))
            pad = 0.05 * (hi - lo if hi > lo else max(abs(hi), 1.0))
            lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y = x")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"Endpoint predictions  ({split} split)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {out_path}")


def plot_endpoints_by_source(result: dict, protein_sp: str, mrna_sp: str,
                             split: str, out_path: Path) -> None:
    """Scatter plot coloured by data source (old=blue, new=orange)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = result.get("sources")
    if sources is None or not np.any(sources >= 0):
        return  # no source info available

    src_styles = {0: ("old", "#1f77b4"), 1: ("new", "#ff7f0e")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    panels = [
        (axes[0], "true_protein_final", "pred_protein_final",
         f"Protein final yield ({protein_sp})"),
        (axes[1], "true_mrna_max", "pred_mrna_max",
         f"mRNA max ({mrna_sp})"),
    ]

    for ax, tkey, pkey, title in panels:
        all_vals = []
        for src_id, (src_name, color) in src_styles.items():
            mask = sources == src_id
            if mask.sum() < 2:
                continue
            t, p = result[tkey][mask], result[pkey][mask]
            r2_val = r2(t, p)
            ax.scatter(t, p, s=30, alpha=0.8, color=color,
                       edgecolors="white", linewidths=0.4,
                       label=f"{src_name}  R²={r2_val:.3f}  (n={mask.sum()})")
            all_vals.extend(t); all_vals.extend(p)

        # overall R² line
        t_all = result[tkey][sources >= 0]
        p_all = result[pkey][sources >= 0]
        if len(t_all) >= 2:
            r2_all = r2(t_all, p_all)
            ax.set_title(f"{title}\noverall R²={r2_all:.3f}")

        if all_vals:
            lo, hi = float(min(all_vals)), float(max(all_vals))
            pad = 0.05 * (hi - lo if hi > lo else max(abs(hi), 1.0))
            lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

        ax.set_xlabel("true"); ax.set_ylabel("predicted")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=9, loc="best")

    fig.suptitle(f"Endpoint predictions by source  ({split} split)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {out_path}")


def r2_by_source(result: dict) -> dict[int, dict[str, float]]:
    """Return {src_id: {r2_protein, r2_mrna, n}} for each source present."""
    sources = result.get("sources")
    if sources is None:
        return {}
    out = {}
    for src_id in np.unique(sources[sources >= 0]):
        mask = sources == src_id
        if mask.sum() < 2:
            continue
        out[int(src_id)] = {
            "r2_protein": float(r2(result["true_protein_final"][mask],
                                   result["pred_protein_final"][mask])),
            "r2_mrna":    float(r2(result["true_mrna_max"][mask],
                                   result["pred_mrna_max"][mask])),
            "n":          int(mask.sum()),
        }
    return out


def save_r2_cache(exp_dir: Path, result: dict, r2_protein: float, r2_mrna: float) -> None:
    import csv
    by_src = r2_by_source(result)
    src_names = {0: "old", 1: "new"}
    fieldnames = ["run", "n", "r2_protein_final", "r2_mrna_max"]
    for src_id, src_name in src_names.items():
        if src_id in by_src:
            fieldnames += [f"r2_protein_{src_name}", f"r2_mrna_{src_name}", f"n_{src_name}"]
    cache = exp_dir / "r2_cache.csv"
    with open(cache, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        row = {
            "run": exp_dir.name,
            "n": result["n"],
            "r2_protein_final": f"{r2_protein:.8f}",
            "r2_mrna_max": f"{r2_mrna:.8f}",
        }
        for src_id, src_name in src_names.items():
            if src_id in by_src:
                s = by_src[src_id]
                row[f"r2_protein_{src_name}"] = f"{s['r2_protein']:.8f}"
                row[f"r2_mrna_{src_name}"]    = f"{s['r2_mrna']:.8f}"
                row[f"n_{src_name}"]           = s["n"]
        writer.writerow(row)


def print_table(results: list[dict], protein_sp: str, mrna_sp: str) -> None:
    print()
    print(f"{'run':<55}  {'n':>4}  {'R²(protein final)':>18}  {'R²(mRNA max)':>14}")
    print("-" * 100)
    for res in results:
        r2_p = r2(res["true_protein_final"], res["pred_protein_final"])
        r2_m = r2(res["true_mrna_max"],      res["pred_mrna_max"])
        print(f"{res['run']:<55}  {res['n']:>4}  {r2_p:>18.4f}  {r2_m:>14.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Experiment folder (many runs) or a single run dir")
    parser.add_argument("--split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--protein", default="pm", help="Protein species name (default: pm)")
    parser.add_argument("--mrna",    default="mm", help="mRNA species name (default: mm)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path (default: <root>/endpoint_r2.png)")
    args = parser.parse_args()

    root = Path(args.root)
    runs = _find_runs(root)
    if not runs:
        print(f"No completed runs found in {root}")
        sys.exit(1)

    device = device_auto()
    print(f"Device: {device}   split: {args.split}   runs: {len(runs)}")

    results = []
    for exp_dir in runs:
        print(f"  evaluating {exp_dir.name}...")
        try:
            results.append(
                collect_endpoints(exp_dir, device, args.split, args.protein, args.mrna)
            )
        except Exception as e:
            print(f"    skip: {e}")

    if not results:
        print("No runs evaluated.")
        sys.exit(1)

    print_table(results, args.protein, args.mrna)

    out = Path(args.out) if args.out else root / "endpoint_r2.png"
    plot_endpoints(results, args.protein, args.mrna, args.split, out)
