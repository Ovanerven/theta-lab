"""
Paper plot: small-model vs big-model theta dynamics & trajectories.

Hypothesis we want to make visible: a small (under-parameterised) scaffold
has to "do more work" through its theta variations, whereas a larger scaffold
sits closer to a constant theta — so theta(t) should look noticeably flatter
in the big-model column.

For each system we pick two scaffolds (small / big) and produce a 2x2
figure:
        col 0 (small)         col 1 (big)
row 0   theta(t)              theta(t)         <- shared y axis
row 1   trajectory(t)         trajectory(t)

Clean style for paper: no background grid, only left/bottom spines, small
fonts, sized for a NeurIPS column.

Run on a slurm node:
    sbatch slurm_jobs/plot_paper_theta_compare.job
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_paper_grid import (
    REPO,
    EXP,
    pick_test_positions,
    run_inference_for_indices,
    _slug,
)
from plot_diagnostics import (
    rebuild_model_from_experiment,
    device_auto,
    load_yaml,
    _filter_model_kwargs,
    _test_subset,
)
from train import collate, collate_varlen


# (small_label, small_exp, big_label, big_exp)
SCENARIOS = {
    "MOF": (
        "MOF / 4-state",  EXP / "mof_synthesis_4_data_ablation"  / "20260502_150937_A6_baseline_n1000",
        "MOF / 12-state", EXP / "mof_synthesis_12_data_ablation" / "20260505_183402_A7_l1reg_n1000",
    ),
    "Enzyme": (
        "Enzyme / 2-state", EXP / "single_enzyme_lumped_data_ablation" / "20260502_151643_A6_baseline_n1000",
        "Enzyme / 6-state", EXP / "single_enzyme_6_data_ablation"      / "20260505_133623_A7_l1reg_n1000",
    ),
    "Glycolysis": (
        "Glycolysis / 4-state",  EXP / "glycolysis_reduced4_data_ablation" / "20260505_023727_A6_baseline_n1000",
        "Glycolysis / 22-state", EXP / "glycolysis_full22_data_ablation"   / "20260506_012257_A7_l1reg_n1000",
    ),
}

RNG_SEED = 0


def _shared_raw_idx(small_exp: Path, big_exp: Path, rng: np.random.Generator) -> int:
    """Pick a raw dataset index present in both runs' test splits.
    Falls back to the small run's first test index if the big run has no split."""
    def test_idx(exp_dir: Path) -> list[int] | None:
        sp = exp_dir / "split.npz"
        if not sp.exists():
            return None
        return np.load(sp)["test_idx"].tolist()

    a = test_idx(small_exp)
    b = test_idx(big_exp)
    if a is None and b is None:
        return 0
    if a is None:
        return int(b[0])
    if b is None:
        return int(a[0])
    common = sorted(set(a) & set(b))
    if not common:
        print(f"  [warn] no shared test idx between {small_exp.name} and {big_exp.name}; "
              f"using {a[0]} (small) — trajectories will differ")
        return int(a[0])
    return int(common[0])


def _position_of_raw_idx(exp_dir: Path, raw_idx: int) -> int:
    """Convert a raw dataset index into a position within the test subset."""
    sp = exp_dir / "split.npz"
    if not sp.exists():
        return raw_idx
    test_idx = np.load(sp)["test_idx"].tolist()
    if raw_idx in test_idx:
        return test_idx.index(raw_idx)
    return 0


def run_with_theta(model, ds, exp_dir, device, position):
    """Forward pass, return (t, y_true, y_pred, theta) for a single test position."""
    raw_ds = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    plot_ds = _test_subset(ds, exp_dir)

    collate_fn = collate_varlen if getattr(raw_ds, "variable_length", False) else collate
    loader = torch.utils.data.DataLoader(
        plot_ds, batch_size=len(plot_ds), shuffle=False, num_workers=0, collate_fn=collate_fn
    )
    y0, u_seq, y_seq, batch_lengths = next(iter(loader))
    K = u_seq.shape[1]
    dt_seq = torch.from_numpy(raw_ds.dt[:K])[None, :].expand(y0.shape[0], -1)
    y0, u_seq, y_seq, dt_seq = (t.to(device) for t in (y0, u_seq, y_seq, dt_seq))
    if batch_lengths is not None:
        batch_lengths = batch_lengths.to(device)

    cfg = load_yaml(exp_dir / "config.yaml")
    with torch.no_grad():
        obs_idx = torch.arange(y0.shape[-1], device=y0.device)
        kwargs = {"y_seq": None, "teacher_forcing": False,
                  "u_transform": str(cfg.get("u_transform", "none")),
                  "y_transform": str(cfg.get("y_transform", "none"))}
        if model.__class__.__name__ == "OdeTransformerGrouped" and batch_lengths is not None:
            kwargs["lengths"] = batch_lengths
        pred, theta, _ = model(y0, u_seq, dt_seq, obs_idx,
                               **_filter_model_kwargs(model, kwargs))

    L = int(batch_lengths[position].item()) if batch_lengths is not None else u_seq.shape[1]
    dt_np = dt_seq[position, :L].cpu().numpy()
    t = np.cumsum(dt_np)
    y_true = y_seq[position, :L].cpu().numpy()
    y_pred = pred[position, :L].cpu().numpy()
    theta_np = theta[position, :L].cpu().numpy() if theta is not None else None
    state_names = (raw_ds.obs_names.tolist()
                   if getattr(raw_ds, "obs_names", None) is not None
                   else [f"y{i}" for i in range(y_seq.shape[-1])])
    return t, y_true, y_pred, theta_np, state_names


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=7)
    ax.margins(x=0)


def plot_scenario(system_name: str, small_label: str, small_exp: Path,
                  big_label: str, big_exp: Path, out_dir: Path):
    device = device_auto()
    rng = np.random.default_rng(RNG_SEED)

    raw_idx = _shared_raw_idx(small_exp, big_exp, rng)
    print(f"  using shared raw idx={raw_idx}")

    runs = []
    for label, exp_dir in [(small_label, small_exp), (big_label, big_exp)]:
        model, ds, _, _ = rebuild_model_from_experiment(exp_dir, device)
        pos = _position_of_raw_idx(exp_dir, raw_idx)
        runs.append((label, *run_with_theta(model, ds, exp_dir, device, pos)))

    # Normalize each theta_d by its time-mean so all params hover around 1.0.
    # Lets us compare *relative* variation across scaffolds with different
    # parameter magnitudes on a shared y-axis.
    runs_norm = []
    for r in runs:
        label, t, y_true, y_pred, theta_np, state_names = r
        if theta_np is not None:
            mean = np.mean(theta_np, axis=0, keepdims=True)
            mean = np.where(np.abs(mean) < 1e-9, 1.0, mean)
            theta_np = theta_np / mean
        runs_norm.append((label, t, y_true, y_pred, theta_np, state_names))
    runs = runs_norm

    theta_min = min(np.nanmin(r[4]) for r in runs if r[4] is not None)
    theta_max = max(np.nanmax(r[4]) for r in runs if r[4] is not None)
    pad = 0.05 * (theta_max - theta_min + 1e-9)
    theta_ylim = (theta_min - pad, theta_max + pad)

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 3.4),
                             gridspec_kw={"height_ratios": [1.0, 1.0]},
                             sharex="col")

    for c, (label, t, y_true, y_pred, theta_np, state_names) in enumerate(runs):
        ax_t = axes[0, c]
        ax_y = axes[1, c]

        # theta panel — fade individual lines, draw mean bold
        if theta_np is not None:
            D = theta_np.shape[-1]
            cmap = plt.cm.viridis(np.linspace(0.0, 0.85, D))
            for d in range(D):
                ax_t.plot(t, theta_np[:, d], color=cmap[d], lw=0.7, alpha=0.85)
        ax_t.set_ylim(*theta_ylim)
        ax_t.set_title(label, fontsize=9)
        ax_t.axhline(1.0, color="k", lw=0.5, alpha=0.4, linestyle=":")
        if c == 0:
            ax_t.set_ylabel(r"$\theta(t) / \overline{\theta}$", fontsize=9)
        else:
            ax_t.tick_params(labelleft=False)
        _clean_axes(ax_t)

        # trajectory panel — all states overlay, true solid, pred dashed
        P = y_true.shape[-1]
        cmap_y = plt.cm.tab10(np.arange(P) % 10)
        for p in range(P):
            ax_y.plot(t, y_true[:, p], color=cmap_y[p], lw=1.0, alpha=0.9)
            ax_y.plot(t, y_pred[:, p], color=cmap_y[p], lw=1.0, alpha=0.9, linestyle="--")
        if c == 0:
            ax_y.set_ylabel("states", fontsize=9)
        ax_y.set_xlabel("time", fontsize=9)
        _clean_axes(ax_y)

    # legend for true vs pred only (states colours are not interpreted)
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], color="black", lw=1.0, label="true"),
        Line2D([0], [0], color="black", lw=1.0, linestyle="--", label="pred"),
    ]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.03, 1, 1.0])

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = out_dir / f"theta_compare_{system_name.lower()}.{ext}"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"wrote {out_path}")
    plt.close(fig)


def main():
    out_dir = REPO / "results" / "paper_grids" / "theta_compare"
    for sys_name, payload in SCENARIOS.items():
        small_label, small_exp, big_label, big_exp = payload
        if not (small_exp.exists() and big_exp.exists()):
            print(f"[skip] {sys_name}: missing experiment dirs")
            continue
        print(f"[{sys_name}] small={small_exp.name} | big={big_exp.name}")
        plot_scenario(sys_name, small_label, small_exp, big_label, big_exp, out_dir)


if __name__ == "__main__":
    main()
