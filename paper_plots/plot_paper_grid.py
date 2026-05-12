"""
Compact N x 3 grid plot for the paper.

For each system (MOF, Enzyme, Glycolysis), build one figure where:
    rows    = scaffolds (e.g. MOF/4-state, MOF/6-state, ...)
    columns = 3 test samples
    each cell overlays true (solid) vs pred (dashed) on all observed states.

Uses the A6 baseline n=1000 run for every scaffold.

Run:
    cd last-layer-ode
    python plot_paper_grid.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_diagnostics import (
    rebuild_model_from_experiment,
    device_auto,
    load_yaml,
    _filter_model_kwargs,
    _test_subset,
)
from train import collate, collate_varlen


REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"


SYSTEMS = {
    "MOF": [
        ("MOF / 4-state",  EXP / "mof_synthesis_4_data_ablation"  / "20260502_150937_A6_baseline_n1000"),
        ("MOF / 6-state",  EXP / "mof_synthesis_6_data_ablation"  / "20260505_095908_A6_baseline_n1000"),
        ("MOF / 8-state",  EXP / "mof_synthesis_8_data_ablation"  / "20260505_095919_A6_baseline_n1000"),
        ("MOF / 12-state", EXP / "mof_synthesis_12_data_ablation" / "20260505_183344_A6_baseline_n1000"),
    ],
    "Enzyme": [
        ("Enzyme / 2-state", EXP / "single_enzyme_lumped_data_ablation" / "20260502_151643_A6_baseline_n1000"),
        ("Enzyme / 4-state", EXP / "single_enzyme_4_data_ablation"      / "20260505_011747_A6_baseline_n1000"),
        ("Enzyme / 6-state", EXP / "single_enzyme_6_data_ablation"      / "20260505_130607_A6_baseline_n1000"),
    ],
    "Glycolysis": [
        ("Glycolysis / 22-state", EXP / "glycolysis_full22_data_ablation"    / "20260506_012257_A7_l1reg_n1000"),
        ("Glycolysis / 12-state", EXP / "glycolysis_reduced12_data_ablation" / "20260505_095854_A6_baseline_n1000"),
        ("Glycolysis / 8-state",  EXP / "glycolysis_reduced8_data_ablation"  / "20260505_012446_A6_baseline_n1000"),
        ("Glycolysis / 4-state",  EXP / "glycolysis_reduced4_data_ablation"  / "20260505_023727_A6_baseline_n1000"),
    ],
}

N_SAMPLES = 3
RNG_SEED = 0


def run_inference_for_indices(model, ds, exp_dir, device, sample_positions):
    """Run a forward pass over the test split and return per-sample (t, y_true, y_pred).

    sample_positions: positions within the test subset (0..len(test)-1) to extract.
    """
    raw_ds = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    plot_ds = _test_subset(ds, exp_dir)

    collate_fn = collate_varlen if getattr(raw_ds, "variable_length", False) else collate
    loader = torch.utils.data.DataLoader(
        plot_ds, batch_size=len(plot_ds), shuffle=False, num_workers=0, collate_fn=collate_fn
    )
    y0, u_seq, y_seq, batch_lengths = next(iter(loader))
    K_batch = u_seq.shape[1]
    dt_seq = torch.from_numpy(raw_ds.dt[:K_batch])[None, :].expand(y0.shape[0], -1)

    y0, u_seq, y_seq, dt_seq = (t.to(device) for t in (y0, u_seq, y_seq, dt_seq))
    if batch_lengths is not None:
        batch_lengths = batch_lengths.to(device)

    cfg_local = load_yaml(exp_dir / "config.yaml")
    with torch.no_grad():
        obs_idx = torch.arange(y0.shape[-1], device=y0.device)
        model_kwargs = {
            "y_seq": None, "teacher_forcing": False,
            "u_transform": str(cfg_local.get("u_transform", "none")),
            "y_transform": str(cfg_local.get("y_transform", "none")),
        }
        if model.__class__.__name__ == "OdeTransformerGrouped" and batch_lengths is not None:
            model_kwargs["lengths"] = batch_lengths
        pred, _theta, _beta = model(y0, u_seq, dt_seq, obs_idx,
                                    **_filter_model_kwargs(model, model_kwargs))

    out = []
    for pos in sample_positions:
        Li = int(batch_lengths[pos].item()) if batch_lengths is not None else u_seq.shape[1]
        dt_np = dt_seq[pos, :Li].cpu().numpy()
        t = np.cumsum(dt_np)
        y_true = y_seq[pos, :Li].cpu().numpy()
        y_pred = pred[pos, :Li].cpu().numpy()
        out.append((t, y_true, y_pred))

    state_names = (raw_ds.obs_names.tolist()
                   if getattr(raw_ds, "obs_names", None) is not None
                   else [f"y{i}" for i in range(y_seq.shape[-1])])
    return out, state_names


def pick_test_positions(exp_dir: Path, n: int, rng: np.random.Generator) -> list[int]:
    split_path = exp_dir / "split.npz"
    if split_path.exists():
        split = np.load(split_path)
        n_test = len(split["test_idx"])
    else:
        n_test = n  # fallback
    if n_test <= n:
        return list(range(n_test))
    return sorted(rng.choice(n_test, size=n, replace=False).tolist())


def plot_system(system_name: str, scaffolds: list[tuple[str, Path]], out_dir: Path):
    device = device_auto()
    rng = np.random.default_rng(RNG_SEED)
    N = len(scaffolds)

    # Compact NeurIPS-friendly figure: ~textwidth wide, modest height per row.
    fig_w = 6.5
    fig_h = 1.55 * N + 0.4
    fig, axes = plt.subplots(N, N_SAMPLES, figsize=(fig_w, fig_h),
                             squeeze=False, sharex=False)

    for r, (label, exp_dir) in enumerate(scaffolds):
        if not exp_dir.exists():
            print(f"[skip] {exp_dir} missing")
            for c in range(N_SAMPLES):
                axes[r, c].axis("off")
            continue
        print(f"[{system_name}] {label}: {exp_dir.name}")
        model, ds, _state_names_cfg, _ = rebuild_model_from_experiment(exp_dir, device)
        positions = pick_test_positions(exp_dir, N_SAMPLES, rng)
        results, state_names = run_inference_for_indices(model, ds, exp_dir, device, positions)

        for c, (t, y_true, y_pred) in enumerate(results):
            ax = axes[r, c]
            P = y_true.shape[-1]
            for p in range(P):
                color = plt.cm.tab10(p % 10)
                ax.plot(t, y_true[:, p], color=color, lw=1.0, alpha=0.9)
                ax.plot(t, y_pred[:, p], color=color, lw=1.0, alpha=0.9, linestyle="--")
            ax.tick_params(axis="both", labelsize=7)
            ax.margins(x=0)
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
            if r == 0:
                ax.set_title(f"sample {c+1}", fontsize=8)
            if r == N - 1:
                ax.set_xlabel("time", fontsize=8)

    # one combined legend at the bottom
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="black", lw=1.2, linestyle="-",  label="true"),
        Line2D([0], [0], color="black", lw=1.2, linestyle="--", label="pred"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.02, 1, 1.0])

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = out_dir / f"paper_grid_{system_name.lower()}.{ext}"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"  wrote {out_path}")
    plt.close(fig)


def _slug(s: str) -> str:
    return (s.lower().replace(" / ", "_").replace("/", "_")
              .replace(" ", "").replace("-", ""))


def plot_per_sample(system_name: str, scaffolds: list[tuple[str, Path]], out_dir: Path):
    """One small figure per (scaffold, sample): subplots = species, true vs pred."""
    device = device_auto()
    rng = np.random.default_rng(RNG_SEED)
    sys_dir = out_dir / _slug(system_name)
    sys_dir.mkdir(parents=True, exist_ok=True)

    for label, exp_dir in scaffolds:
        if not exp_dir.exists():
            print(f"[skip] {exp_dir} missing")
            continue
        print(f"[per-sample][{system_name}] {label}: {exp_dir.name}")
        model, ds, _, _ = rebuild_model_from_experiment(exp_dir, device)
        positions = pick_test_positions(exp_dir, N_SAMPLES, rng)
        results, state_names = run_inference_for_indices(model, ds, exp_dir, device, positions)

        for s_idx, (t, y_true, y_pred) in enumerate(results):
            P = y_true.shape[-1]
            ncols = min(P, 4)
            nrows = int(np.ceil(P / ncols))
            fig_w = 1.7 * ncols + 0.4
            fig_h = 1.45 * nrows + 0.4
            fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                                     squeeze=False, sharex=True)
            for p in range(nrows * ncols):
                r, c = divmod(p, ncols)
                ax = axes[r, c]
                if p >= P:
                    ax.axis("off")
                    continue
                ax.plot(t, y_true[:, p], color="C0", lw=1.2, label="true")
                ax.plot(t, y_pred[:, p], color="C1", lw=1.2, linestyle="--", label="pred")
                name = state_names[p] if p < len(state_names) else f"y{p}"
                ax.set_title(name, fontsize=8)
                ax.tick_params(axis="both", labelsize=7)
                ax.margins(x=0)
                if r == nrows - 1:
                    ax.set_xlabel("time", fontsize=8)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center",
                       bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8, frameon=False)
            fig.tight_layout(rect=[0, 0.02, 1, 1.0])

            tag = f"{_slug(label)}_sample{s_idx+1}"
            for ext in ("png", "pdf"):
                out_path = sys_dir / f"paper_sample_{tag}.{ext}"
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"  wrote {sys_dir / ('paper_sample_' + tag + '.{png,pdf}')}")
            plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("grid", "per-sample", "both"), default="both")
    args = ap.parse_args()

    out_dir = REPO / "results" / "paper_grids"
    for sys_name, scaffolds in SYSTEMS.items():
        if args.mode in ("grid", "both"):
            plot_system(sys_name, scaffolds, out_dir)
        if args.mode in ("per-sample", "both"):
            plot_per_sample(sys_name, scaffolds, out_dir)


if __name__ == "__main__":
    main()
