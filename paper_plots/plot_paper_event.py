"""
Paper-quality event plot: θ(t) on top, states (true vs pred) on bottom.

Generates 4 styling variants for choosing the cleanest one for the appendix:
  v1_dashed     : vertical dashed colored lines through every panel (current style)
  v2_topticks   : event marks only above the θ row as colored triangles, panels stay clean
  v3_bands      : narrow shaded vertical bands at event times (faint)
  v4_split_top  : event lines on θ row only, trajectory row is unmarked

Usage:
    python plot_paper_event.py \
        --exp experiments/single_enzyme_lumped_data_ablation/20260502_151710_A8_l2reg_n1000 \
        --idx 916 --theta 0,1 --states 0,1 --tag lumped_idx916

    python plot_paper_event.py \
        --exp experiments/mof_synthesis_8_data_ablation/20260505_095919_A6_baseline_n1000 \
        --idx 916 --theta auto:4 --states auto --tag mof8_idx916
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from plot_diagnostics import (
    rebuild_model_from_experiment,
    device_auto,
    load_yaml,
    _filter_model_kwargs,
    _test_subset,
)
from train import collate, collate_varlen


def _run_inference(model, ds, exp_dir, device, raw_idx):
    """Run model forward on a single raw-dataset index. Returns t, y_true, y_pred, u, theta, control_names."""
    raw_ds = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    plot_ds = _test_subset(ds, exp_dir)
    if isinstance(plot_ds, torch.utils.data.Subset):
        all_indices = list(plot_ds.indices)
    else:
        all_indices = list(range(len(plot_ds)))
    if raw_idx not in all_indices:
        raise SystemExit(f"raw idx {raw_idx} is not in the test split for this experiment. "
                         f"Available (first 20): {all_indices[:20]}")
    pos = all_indices.index(raw_idx)

    collate_fn = collate_varlen if getattr(raw_ds, "variable_length", False) else collate
    loader = torch.utils.data.DataLoader(plot_ds, batch_size=len(plot_ds), shuffle=False,
                                         num_workers=0, collate_fn=collate_fn)
    y0, u_seq, y_seq, batch_lengths = next(iter(loader))
    K_batch = u_seq.shape[1]
    dt_seq = torch.from_numpy(raw_ds.dt[:K_batch])[None, :].expand(y0.shape[0], -1)

    y0, u_seq, y_seq, dt_seq = (t.to(device) for t in (y0, u_seq, y_seq, dt_seq))
    if batch_lengths is not None:
        batch_lengths = batch_lengths.to(device)

    with torch.no_grad():
        obs_idx = torch.arange(y0.shape[-1], device=y0.device)
        cfg_local = load_yaml(exp_dir / "config.yaml")
        model_kwargs = {"y_seq": None, "teacher_forcing": False,
                        "u_transform": str(cfg_local.get("u_transform", "none")),
                        "y_transform": str(cfg_local.get("y_transform", "none"))}
        if model.__class__.__name__ == "OdeTransformerGrouped" and batch_lengths is not None:
            model_kwargs["lengths"] = batch_lengths
        pred, theta, _beta = model(y0, u_seq, dt_seq, obs_idx,
                                   **_filter_model_kwargs(model, model_kwargs))

    Li = int(batch_lengths[pos].item()) if batch_lengths is not None else u_seq.shape[1]
    dt_np = dt_seq[pos, :Li].cpu().numpy()
    t = np.cumsum(dt_np)
    y_true = y_seq[pos, :Li].cpu().numpy()
    y_pred = pred[pos, :Li].cpu().numpy()
    u_np = u_seq[pos, :Li].cpu().numpy()
    theta_np = theta[pos, :Li].cpu().numpy() if theta is not None else None

    if getattr(raw_ds, "control_names", None) is not None:
        control_names = raw_ds.control_names.tolist()
    else:
        control_names = [f"u{j}" for j in range(u_np.shape[-1])]

    return t, y_true, y_pred, u_np, theta_np, control_names


def _events(t, u_np, thresh=1e-4):
    """Return list of (time, control_idx, magnitude)."""
    out = []
    U = u_np.shape[-1]
    for j in range(U):
        for k in np.where(u_np[:, j] > thresh)[0]:
            out.append((float(t[k]), j, float(u_np[k, j])))
    return out


def _add_magnitude_labels(ax, events, fmt="{:.2g}"):
    """Annotate each event with its bolus magnitude just above the panel."""
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    yt = ymax + 0.02 * yspan
    for et, j, mag in events:
        c = f"C{j % 10}"
        ax.text(et, yt, fmt.format(mag),
                color=c, fontsize=8, ha="center", va="bottom",
                clip_on=False, zorder=6,
                fontweight="bold")
    ax.set_ylim(ymin, ymax + 0.18 * yspan)


def _select_theta_indices(theta_np, spec):
    """Resolve --theta spec. 'auto:K' picks K most-variable θs (by std)."""
    D = theta_np.shape[-1]
    if spec.startswith("auto"):
        k = int(spec.split(":")[1]) if ":" in spec else min(4, D)
        # rank by std of d(theta)/dt-like response: use total absolute change
        score = np.std(theta_np, axis=0)
        order = np.argsort(-score)
        return sorted(order[:k].tolist())
    return [int(x) for x in spec.split(",")]


def _select_state_indices(P, spec, y_true=None):
    if spec is None or spec == "auto":
        return list(range(P))
    if spec.startswith("auto:") and y_true is not None:
        k = min(int(spec.split(":")[1]), P)
        score = np.std(y_true, axis=0)
        order = np.argsort(-score)
        return sorted(order[:k].tolist())
    return [int(x) for x in spec.split(",")]


def _draw_panel(ax, t, y, color, lw=2.0, label=None, ls="-"):
    ax.plot(t, y, color=color, linewidth=lw, label=label, linestyle=ls)
    ax.grid(True, alpha=0.25)
    ax.set_xlim([t[0], t[-1]])


def _add_events(ax, events, style, ymin_frac=0.0, ymax_frac=1.0):
    """style ∈ {dashed, bands, none}. ymin/ymax: fraction of axis height used by markers."""
    if style == "none":
        return
    for et, j, _ in events:
        c = f"C{j % 10}"
        if style == "dashed":
            ax.axvline(et, color=c, linestyle="--", alpha=0.55, linewidth=1.3)
        elif style == "bands":
            ax.axvspan(et - 0.05, et + 0.05, color=c, alpha=0.18, linewidth=0)


def _add_top_ticks(ax, events, control_names):
    """Place small colored triangles just above the top of `ax`, in axis coords."""
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    yt = ymax + 0.04 * yspan
    for et, j, _ in events:
        c = f"C{j % 10}"
        ax.plot([et], [yt], marker="v", color=c, markersize=8, clip_on=False, zorder=5)
    # extend ylim slightly so triangles fit
    ax.set_ylim(ymin, ymax + 0.10 * yspan)


def _make_figure(t, y_true, y_pred, u_np, theta_np,
                 state_names, control_names, param_names,
                 theta_idx, state_idx, style, title, out_path,
                 show_magnitudes=True):
    D = len(theta_idx)
    P = len(state_idx)
    ncols = max(D, P)
    events = _events(t, u_np)

    fig_w = max(8, 2.8 * ncols)
    fig, axes = plt.subplots(2, ncols, figsize=(fig_w, 5.6), sharex=True,
                             squeeze=False)

    # Top row: theta
    for c in range(ncols):
        ax = axes[0, c]
        if c < D:
            d = theta_idx[c]
            name = (param_names[d] if param_names and d < len(param_names) else f"θ{d}")
            _draw_panel(ax, t, theta_np[:, d], color="C3", lw=2.0)
            ax.set_title(f"{name}", fontsize=12, fontweight="bold", color="C3")
            # event markers
            if style in ("v1_dashed",):
                _add_events(ax, events, "dashed")
            elif style == "v3_bands":
                _add_events(ax, events, "bands")
            elif style == "v4_split_top":
                _add_events(ax, events, "dashed")
            # v2: top ticks added after ylim is finalized below
        else:
            ax.axis("off")

    # Bottom row: states
    for c in range(ncols):
        ax = axes[1, c]
        if c < P:
            p = state_idx[c]
            name = state_names[p] if p < len(state_names) else f"y{p}"
            _draw_panel(ax, t, y_true[:, p], color="C0", lw=2.0, label="true")
            _draw_panel(ax, t, y_pred[:, p], color="C1", lw=2.0, label="pred", ls="--")
            ax.set_title(f"{name}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Time", fontsize=10)
            if c == 0:
                ax.legend(fontsize=9, loc="best", frameon=True)
            if style == "v1_dashed":
                _add_events(ax, events, "dashed")
            elif style == "v3_bands":
                _add_events(ax, events, "bands")
            # v2 top-ticks: add to top row only — leave traj clean
            # v4 split_top: leave bottom row clean
        else:
            ax.axis("off")

    # v2 top-ticks pass: applied last so ylim is computed
    if style == "v2_topticks":
        for c in range(D):
            _add_top_ticks(axes[0, c], events, control_names)

    # Magnitude labels on top (θ) row only — keeps trajectory row uncluttered.
    if show_magnitudes:
        for c in range(D):
            _add_magnitude_labels(axes[0, c], events)

    # Legend for events
    custom = [Line2D([0], [0], color=f"C{j % 10}", lw=2,
                     linestyle=("--" if style != "v3_bands" else "-"),
                     marker=("v" if style == "v2_topticks" else None),
                     markersize=8)
              for j in range(len(control_names))]
    fig.legend(custom, control_names, loc="upper center",
               bbox_to_anchor=(0.5, 0.99),
               ncol=min(len(control_names), 6),
               title="Bolus Events", fontsize=9, title_fontsize=10,
               frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, type=Path)
    ap.add_argument("--idx", required=True, type=int, help="raw dataset index (e.g. 916)")
    ap.add_argument("--theta", default="auto:2",
                    help="comma list of θ indices, or 'auto:K' for top-K by variance")
    ap.add_argument("--states", default="auto",
                    help="comma list of state indices, or 'auto' for all")
    ap.add_argument("--tag", default=None, help="filename tag; defaults to <expname>_idx<idx>")
    ap.add_argument("--out", default=None,
                    help="output dir; defaults to <exp>/plots/")
    ap.add_argument("--styles", default="v1_dashed,v2_topticks,v3_bands,v4_split_top")
    ap.add_argument("--no-magnitudes", action="store_true",
                    help="hide bolus magnitude labels (default: shown above θ panels)")
    args = ap.parse_args()

    device = device_auto()
    exp_dir = args.exp.resolve()
    model, ds, state_names, param_names = rebuild_model_from_experiment(exp_dir, device)

    t, y_true, y_pred, u_np, theta_np, control_names = _run_inference(
        model, ds, exp_dir, device, raw_idx=args.idx)
    if theta_np is None:
        raise SystemExit("Model returned no theta — cannot build paper event plot.")

    theta_idx = _select_theta_indices(theta_np, args.theta)
    state_idx = _select_state_indices(y_pred.shape[-1], args.states, y_true=y_true)

    out_dir = Path(args.out) if args.out else (exp_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{exp_dir.name}_idx{args.idx}"

    title = f"{exp_dir.parent.name} | {exp_dir.name} | sample idx={args.idx}"
    print(f"theta_idx={theta_idx} ({[param_names[i] if i < len(param_names) else i for i in theta_idx]})")
    print(f"state_idx={state_idx} ({[state_names[i] if i < len(state_names) else i for i in state_idx]})")
    print(f"events: {[(round(et,2), control_names[j], round(m,3)) for et,j,m in _events(t, u_np)]}")

    for style in args.styles.split(","):
        for ext in ("png", "pdf"):
            out_path = out_dir / f"paper_event_{tag}_{style}.{ext}"
            _make_figure(t, y_true, y_pred, u_np, theta_np,
                         state_names, control_names, param_names,
                         theta_idx, state_idx, style, title, out_path,
                         show_magnitudes=not args.no_magnitudes)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
