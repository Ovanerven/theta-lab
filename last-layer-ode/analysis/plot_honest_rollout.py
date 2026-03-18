"""
plot_honest_rollout.py

Standalone plotting from CSV exports produced by honest_rollout.py.
No fitting — just reads the exports/ directory and regenerates plots.

Usage:
    python plot_honest_rollout.py \
        --export-dir results_hi10_low1e-5/exports \
        --scaffolds reduced2,reduced3,reduced5,reduced7,reduced9,full13 \
        --show-species A,M \
        --sample-idx 1 \
        --out results_hi10_low1e-5/figures/summary.pdf

    # or just plot everything found in exports/
    python plot_honest_rollout.py \
        --export-dir results_hi10_low1e-5/exports \
        --sample-idx 1 \
        --out results_hi10_low1e-5/figures/summary.pdf

    # control grid layout and format
    python plot_honest_rollout.py \
        --export-dir results_hi10_low1e-5/exports \
        --max-cols 6 \
        --fmt png \
        --out results_hi10_low1e-5/figures/summary.pdf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scaffolds import SCAFFOLDS

FULL_SPECIES = list("ABCDEFGHIJKLM")
SCAFFOLD_ALIASES = {"reduced13": "full13", "full": "full13"}


def normalize_scaffold_name(name: str) -> str:
    return SCAFFOLD_ALIASES.get(name.strip(), name.strip())


def nrmse(pred, true):
    mask = ~np.isnan(true)
    if mask.sum() == 0:
        return np.nan
    rng = true[mask].max() - true[mask].min()
    if rng < 1e-10:
        return np.nan
    return float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)) / rng)


# ─────────────────────────────────────────────────────────────────────────────
#  Load from CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_scaffold_csv(export_dir: Path, scaffold_name: str, sample_idx: int) -> dict:
    """
    Load predictions and losses CSVs for one scaffold.
    Returns dict with keys: time, state_names, true, onestep, rollout,
                            loss_onestep, loss_rollout
    or None if files not found.
    """
    sc_dir = export_dir / scaffold_name
    pred_file = sc_dir / f"predictions_sample{sample_idx}.csv"
    loss_file = sc_dir / f"losses_sample{sample_idx}.csv"

    if not pred_file.exists():
        return None

    # parse predictions
    with open(pred_file) as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(pred_file, delimiter=",", skiprows=1)

    time_col = data[:, 0]

    # infer state names from columns like "true_A", "true_M", ...
    true_cols = [h for h in header if h.startswith("true_")]
    state_names = [h.replace("true_", "") for h in true_cols]
    P = len(state_names)

    col_idx = {h: i for i, h in enumerate(header)}
    true_data    = np.column_stack([data[:, col_idx[f"true_{s}"]]    for s in state_names])
    onestep_data = np.column_stack([data[:, col_idx[f"onestep_{s}"]] for s in state_names])
    rollout_data = np.column_stack([data[:, col_idx[f"rollout_{s}"]] for s in state_names])

    result = dict(
        time=time_col,
        state_names=state_names,
        true=true_data,
        onestep=onestep_data,
        rollout=rollout_data,
    )

    # losses (optional)
    if loss_file.exists():
        loss_data = np.loadtxt(loss_file, delimiter=",", skiprows=1)
        result["loss_onestep"] = loss_data[:, 1]
        result["loss_rollout"] = loss_data[:, 2]

    return result


def discover_scaffolds(export_dir: Path, sample_idx: int):
    """Find all scaffold dirs that have a predictions CSV."""
    found = []
    for d in sorted(export_dir.iterdir()):
        if d.is_dir() and (d / f"predictions_sample{sample_idx}.csv").exists():
            found.append(d.name)
    return found


# ─────────────────────────────────────────────────────────────────────────────
#  Summary grid plot
# ─────────────────────────────────────────────────────────────────────────────

def make_summary_plot(loaded, scaffold_order, show_species,
                      sample_idx, out_path, max_cols=6,
                      col_w=3.0, row_h=2.8):
    """
    Grid: rows = (species × scaffold_row_group), columns = scaffolds.
    With max_cols, scaffolds wrap into multiple row-groups so panels
    stay readable.
    """
    n_scaffolds = len(scaffold_order)
    n_species   = len(show_species)

    n_cols = min(n_scaffolds, max_cols)
    n_scaffold_rows = (n_scaffolds + n_cols - 1) // n_cols  # how many row-groups
    n_total_rows = n_species * n_scaffold_rows

    fig, axes = plt.subplots(
        n_total_rows, n_cols,
        figsize=(col_w * n_cols + 0.8, row_h * n_total_rows + 1.2),
        squeeze=False,
    )

    # turn off all axes first, enable as needed
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    for sc_idx, sn in enumerate(scaffold_order):
        col        = sc_idx % n_cols
        row_group  = sc_idx // n_cols  # which row-group (0, 1, ...)
        res        = loaded[sn]
        state_names = res["state_names"]
        tt          = res["time"]

        for sp_idx, sp in enumerate(show_species):
            row = row_group * n_species + sp_idx
            ax  = axes[row][col]
            ax.axis("on")

            if sp not in state_names:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                si = state_names.index(sp)
                gt = res["true"][:, si]
                os = res["onestep"][:, si]
                ro = res["rollout"][:, si]

                ax.plot(tt, gt, lw=2,   color="tab:blue",   label="truth")
                ax.plot(tt, os, lw=1.8, color="tab:orange",  ls="--", label="one-step")
                ax.plot(tt, ro, lw=1.8, color="tab:red",     ls=":",  label="rollout")

                n_os = nrmse(os, gt)
                n_ro = nrmse(ro, gt)
                ax.text(0.97, 0.95,
                        f"OS={n_os:.3f}\nRO={n_ro:.3f}",
                        transform=ax.transAxes, fontsize=7,
                        va="top", ha="right",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

                ax.grid(True, alpha=0.2)

            # titles and labels
            P_sc = SCAFFOLDS[sn].P if sn in SCAFFOLDS else "?"
            if sp_idx == 0:
                ax.set_title(f"{sn}\n(P={P_sc})", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"species {sp}", fontsize=10)
            if sp_idx == n_species - 1:
                ax.set_xlabel("time", fontsize=8)

            # legend only first panel
            if sc_idx == 0 and sp_idx == 0:
                ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        f"Oracle per-step theta fitting  (sample {sample_idx})\n"
        f"Each scaffold uses its own matched dataset  |  "
        f"one-step: from y_true[k]  |  rollout: honest (from own prediction)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary plot     -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  NRMSE vs P line plot (cleanest paper figure)
# ─────────────────────────────────────────────────────────────────────────────

def make_nrmse_vs_P_plot(loaded, scaffold_order, show_species, out_path):
    """
    Clean line plot: x = scaffold size P, y = rollout NRMSE.
    One line per species.  Best figure for the main text.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    species_colors = {"A": "tab:blue", "M": "tab:orange"}

    for sp in show_species:
        ps, nrmses = [], []
        for sn in scaffold_order:
            res = loaded[sn]
            state_names = res["state_names"]
            P_sc = SCAFFOLDS[sn].P if sn in SCAFFOLDS else None
            if P_sc is None or sp not in state_names:
                continue
            si = state_names.index(sp)
            gt = res["true"][:, si]
            n_ro = nrmse(res["rollout"][:, si], gt)
            if not np.isnan(n_ro):
                ps.append(P_sc)
                nrmses.append(n_ro)

        color = species_colors.get(sp, None)
        ax.plot(ps, nrmses, "o-", lw=2, markersize=6, label=f"species {sp}",
                color=color)

    ax.set_xlabel("Scaffold size (P)", fontsize=12)
    ax.set_ylabel("NRMSE (honest rollout)", fontsize=12)
    ax.set_title("Trajectory error vs mechanistic scaffold size", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(set(
        SCAFFOLDS[sn].P for sn in scaffold_order if sn in SCAFFOLDS
    )))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"NRMSE vs P plot  -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  NRMSE bar chart
# ─────────────────────────────────────────────────────────────────────────────

def make_nrmse_plot(loaded, scaffold_order, show_species, out_path):
    n_sp = len(show_species)
    fig, axes = plt.subplots(1, n_sp, figsize=(5 * n_sp, 4), squeeze=False)
    x     = np.arange(len(scaffold_order))
    bar_w = 0.35

    for s_i, sp in enumerate(show_species):
        ax = axes[0][s_i]
        nrmse_os, nrmse_ro, sizes = [], [], []

        for sn in scaffold_order:
            res = loaded[sn]
            state_names = res["state_names"]
            P_sc = SCAFFOLDS[sn].P if sn in SCAFFOLDS else 0
            sizes.append(P_sc)

            if sp in state_names:
                si = state_names.index(sp)
                gt = res["true"][:, si]
                nrmse_os.append(nrmse(res["onestep"][:, si], gt))
                nrmse_ro.append(nrmse(res["rollout"][:, si], gt))
            else:
                nrmse_os.append(np.nan)
                nrmse_ro.append(np.nan)

        ax.bar(x - bar_w / 2, nrmse_os, bar_w, label="one-step",
               color="tab:orange", alpha=0.85)
        ax.bar(x + bar_w / 2, nrmse_ro, bar_w, label="rollout",
               color="tab:red", alpha=0.85)
        ax.set_title(f"NRMSE — species {sp}", fontsize=10)
        ax.set_xlabel("scaffold")
        ax.set_ylabel("NRMSE")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{sn}\n(P={s})" for sn, s in zip(scaffold_order, sizes)],
            fontsize=7, rotation=30, ha="right",
        )
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Oracle per-step theta (honest rollout) — NRMSE\n"
        "(each scaffold uses its own matched dataset)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"NRMSE plot       -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Per-scaffold individual plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_scaffold(loaded, scaffold_order, sample_idx, plot_dir, fmt="pdf"):
    plot_dir.mkdir(parents=True, exist_ok=True)

    for sn in scaffold_order:
        res = loaded[sn]
        state_names = res["state_names"]
        tt = res["time"]
        P = len(state_names)

        out_dir = plot_dir / sn
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── pred vs true ──────────────────────────────────────────────────────
        fig, axes = plt.subplots(P, 1, figsize=(11, max(6, 2.0 * P)), sharex=True)
        if P == 1:
            axes = [axes]

        for p, ax in enumerate(axes):
            ax.plot(tt, res["true"][:, p],    lw=2,   color="tab:blue",   label="truth")
            ax.plot(tt, res["onestep"][:, p], lw=1.8, color="tab:orange", ls="--", label="one-step")
            ax.plot(tt, res["rollout"][:, p], lw=1.8, color="tab:red",    ls=":",  label="rollout")
            ax.set_ylabel(state_names[p])
            ax.grid(True, alpha=0.25)
            if p == 0:
                ax.legend(fontsize=9)

        axes[-1].set_xlabel("Time")
        fig.suptitle(f"Oracle per-step fit — {sn} (sample {sample_idx})")
        fig.tight_layout()
        fig.savefig(out_dir / f"pred_vs_true_{sample_idx:03d}.{fmt}", dpi=150)
        plt.close(fig)

        # ── GD loss ───────────────────────────────────────────────────────────
        if "loss_onestep" in res and "loss_rollout" in res:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.semilogy(tt, res["loss_onestep"], lw=1.5, label="one-step", color="tab:orange")
            ax.semilogy(tt, res["loss_rollout"], lw=1.5, label="rollout",  color="tab:red")
            ax.set_xlabel("Time")
            ax.set_ylabel("GD loss (log)")
            ax.set_title(f"Per-step GD loss at convergence — {sn}")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"gd_losses.{fmt}", dpi=150)
            plt.close(fig)

        print(f"  [{sn}] plots -> {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  Overlay plot (compact — best for papers)
# ─────────────────────────────────────────────────────────────────────────────

def make_overlay_plot(loaded, scaffold_order, show_species,
                      sample_idx, out_path):
    """
    Two panels (one per species in show_species).  Each panel overlays
    rollout trajectories from all scaffolds, color-coded by scaffold size,
    with ground truth in black.  Compact and immediately shows the
    tipping point.
    """
    n_species = len(show_species)
    fig, axes = plt.subplots(n_species, 1, figsize=(12, 3.5 * n_species),
                             sharex=False)
    if n_species == 1:
        axes = [axes]

    # colour map: gradient from red (small P) to blue (large P)
    sizes = [SCAFFOLDS[sn].P if sn in SCAFFOLDS else 0 for sn in scaffold_order]
    min_p, max_p = min(sizes), max(sizes)
    cmap = plt.cm.RdYlGn  # red=bad (small), green=good (large)

    for ax, sp in zip(axes, show_species):
        # find a scaffold that has this species for the ground truth
        truth_plotted = False

        for idx, sn in enumerate(scaffold_order):
            res = loaded[sn]
            state_names = res["state_names"]
            if sp not in state_names:
                continue

            si = state_names.index(sp)
            tt = res["time"]
            gt = res["true"][:, si]
            ro = res["rollout"][:, si]

            # truth — plot once
            if not truth_plotted:
                ax.plot(tt, gt, lw=2.5, color="black", label="truth", zorder=10)
                truth_plotted = True

            # rollout — colour by scaffold size
            P_sc = SCAFFOLDS[sn].P if sn in SCAFFOLDS else 0
            if max_p > min_p:
                frac = (P_sc - min_p) / (max_p - min_p)
            else:
                frac = 0.5
            color = cmap(frac)

            n_ro = nrmse(ro, gt)
            ax.plot(tt, ro, lw=1.5, color=color, alpha=0.85,
                    label=f"{sn} (P={P_sc}, NRMSE={n_ro:.3f})")

        ax.set_ylabel(f"species {sp}", fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, ncol=2, loc="upper left",
                  framealpha=0.8)

    axes[-1].set_xlabel("time", fontsize=11)
    fig.suptitle(
        f"Honest rollout across scaffolds  (sample {sample_idx})\n"
        f"Ground truth in black — rollout coloured by scaffold size "
        f"(red=small, green=large)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Overlay plot     -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Compact grid (rollout-only, no one-step clutter)
# ─────────────────────────────────────────────────────────────────────────────

def make_compact_grid(loaded, scaffold_order, show_species,
                      sample_idx, out_path, max_cols=6,
                      col_w=2.6, row_h=2.2):
    """
    Like make_summary_plot but shows only truth vs rollout (no one-step),
    with tighter spacing.  One-step is always ~perfect — mentioning it in
    the caption is enough.
    """
    n_scaffolds = len(scaffold_order)
    n_species   = len(show_species)

    n_cols = min(n_scaffolds, max_cols)
    n_scaffold_rows = (n_scaffolds + n_cols - 1) // n_cols
    n_total_rows = n_species * n_scaffold_rows

    fig, axes = plt.subplots(
        n_total_rows, n_cols,
        figsize=(col_w * n_cols + 0.5, row_h * n_total_rows + 1.0),
        squeeze=False,
    )

    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    for sc_idx, sn in enumerate(scaffold_order):
        col       = sc_idx % n_cols
        row_group = sc_idx // n_cols
        res       = loaded[sn]
        state_names = res["state_names"]
        tt        = res["time"]

        for sp_idx, sp in enumerate(show_species):
            row = row_group * n_species + sp_idx
            ax  = axes[row][col]
            ax.axis("on")

            if sp not in state_names:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                si = state_names.index(sp)
                gt = res["true"][:, si]
                ro = res["rollout"][:, si]

                ax.plot(tt, gt, lw=1.8, color="tab:blue",  label="truth")
                ax.plot(tt, ro, lw=1.5, color="tab:red", ls=":", label="rollout")

                n_ro = nrmse(ro, gt)
                ax.text(0.97, 0.95,
                        f"{n_ro:.3f}",
                        transform=ax.transAxes, fontsize=8, fontweight="bold",
                        va="top", ha="right",
                        color="tab:red" if n_ro > 0.05 else "tab:green",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7))

                ax.grid(True, alpha=0.15)
                ax.tick_params(labelsize=7)

            P_sc = SCAFFOLDS[sn].P if sn in SCAFFOLDS else "?"
            if sp_idx == 0:
                ax.set_title(f"{sn} (P={P_sc})", fontsize=8, pad=3)
            if col == 0:
                ax.set_ylabel(sp, fontsize=10)
            if sp_idx == n_species - 1:
                ax.set_xlabel("time", fontsize=7)

            if sc_idx == 0 and sp_idx == 0:
                ax.legend(fontsize=6, loc="upper left")

    fig.suptitle(
        f"Honest rollout — truth vs prediction  (sample {sample_idx})\n"
        f"NRMSE annotated per panel  |  one-step oracle fits perfectly "
        f"for all scaffolds (not shown)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Compact grid     -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  NRMSE table (stdout)
# ─────────────────────────────────────────────────────────────────────────────

def print_nrmse_table(loaded, scaffold_order, show_species):
    print("\n" + "=" * 80)
    header = f"{'Scaffold':<28} {'P':>3}"
    for sp in show_species:
        header += f"  OS-{sp:>1}    RO-{sp:>1}"
    print(header)
    print("=" * 80)

    for sn in scaffold_order:
        res = loaded[sn]
        state_names = res["state_names"]
        P_sc = SCAFFOLDS[sn].P if sn in SCAFFOLDS else "?"
        row = f"{sn:<28} {P_sc:>3}"
        for sp in show_species:
            if sp in state_names:
                si = state_names.index(sp)
                gt = res["true"][:, si]
                row += f"  {nrmse(res['onestep'][:, si], gt):.4f}"
                row += f"  {nrmse(res['rollout'][:, si], gt):.4f}"
            else:
                row += "    N/A     N/A"
        print(row)

    print("=" * 80)
    print("OS = one-step (from y_true[k])  |  RO = honest rollout\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot honest_rollout results from CSV exports (no fitting).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--export-dir", type=str, required=True,
                        help="Path to exports/ directory from honest_rollout.py")
    parser.add_argument("--scaffolds", type=str, default=None,
                        help="Comma-separated scaffold names (default: auto-discover).")
    parser.add_argument("--show-species", type=str, default="A,M")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--max-cols", type=int, default=6,
                        help="Max scaffolds per row in summary grid (default 6).")
    parser.add_argument("--col-w", type=float, default=3.0,
                        help="Column width in inches (default 3.0).")
    parser.add_argument("--row-h", type=float, default=2.8,
                        help="Row height in inches (default 2.8).")
    parser.add_argument("--out", type=str, required=True,
                        help="Output path for summary plot.")
    parser.add_argument("--no-individual", action="store_true",
                        help="Skip per-scaffold individual plots.")
    parser.add_argument("--no-grid", action="store_true",
                        help="Skip the full 3-line grid (truth/onestep/rollout).")
    parser.add_argument("--fmt", type=str, default="pdf",
                        help="Output format: pdf, png, svg (default: pdf).")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    if not export_dir.exists():
        print(f"[error] Export dir not found: {export_dir}")
        sys.exit(1)

    # discover or filter scaffolds
    if args.scaffolds:
        raw = [s.strip() for s in args.scaffolds.split(",") if s.strip()]
        scaffold_order = list(dict.fromkeys(
            normalize_scaffold_name(s) for s in raw
        ))
    else:
        scaffold_order = discover_scaffolds(export_dir, args.sample_idx)
        print(f"Auto-discovered {len(scaffold_order)} scaffolds in {export_dir}")

    # load
    loaded = {}
    for sn in scaffold_order:
        res = load_scaffold_csv(export_dir, sn, args.sample_idx)
        if res is None:
            print(f"[warn] No CSV for '{sn}' sample {args.sample_idx} — skipping.")
        else:
            loaded[sn] = res

    scaffold_order = [sn for sn in scaffold_order if sn in loaded]
    if not scaffold_order:
        print("[error] No data loaded.")
        sys.exit(1)

    print(f"Loaded: {', '.join(scaffold_order)}")

    show_species = [s.strip().upper() for s in args.show_species.split(",")
                    if s.strip().upper() in FULL_SPECIES]

    out_path = Path(args.out)
    stem = out_path.stem
    parent = out_path.parent
    fmt = args.fmt.strip(".").lower()

    # NRMSE table
    print_nrmse_table(loaded, scaffold_order, show_species)

    # 1) overlay plot (most compact — best for papers)
    overlay_path = parent / f"{stem}_overlay.{fmt}"
    make_overlay_plot(loaded, scaffold_order, show_species,
                      args.sample_idx, overlay_path)

    # 2) compact grid (rollout-only, no one-step)
    compact_path = parent / f"{stem}_compact.{fmt}"
    make_compact_grid(loaded, scaffold_order, show_species,
                      args.sample_idx, compact_path,
                      max_cols=args.max_cols,
                      col_w=args.col_w, row_h=args.row_h)

    # 3) full 3-line grid (truth + onestep + rollout)
    if not args.no_grid:
        grid_path = parent / f"{stem}_grid.{fmt}"
        make_summary_plot(loaded, scaffold_order, show_species,
                          args.sample_idx, grid_path,
                          max_cols=args.max_cols,
                          col_w=args.col_w, row_h=args.row_h)

    # 4) NRMSE bar chart
    nrmse_path = parent / f"{stem}_nrmse.{fmt}"
    make_nrmse_plot(loaded, scaffold_order, show_species, nrmse_path)

    # 5) NRMSE vs P line plot (cleanest for main text)
    nrmse_p_path = parent / f"{stem}_nrmse_vs_P.{fmt}"
    make_nrmse_vs_P_plot(loaded, scaffold_order, show_species, nrmse_p_path)

    # 6) per-scaffold individual plots
    if not args.no_individual:
        plot_dir = parent / "plots"
        plot_per_scaffold(loaded, scaffold_order, args.sample_idx, plot_dir, fmt=fmt)

    print("\nDone.")


if __name__ == "__main__":
    main()