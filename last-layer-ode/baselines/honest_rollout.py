"""
honest_rollout.py

Oracle per-step theta fitting with honest rollout.

Two modes are computed per scaffold:

  ONE-STEP (oracle):
    At step k, start from y_true[k], optimise theta, predict y_hat[k+1].
    This is the best-case single-transition fit — errors never compound.

  HONEST ROLLOUT:
    At step k, start from y_current (own previous prediction), optimise
    theta to hit y_true[k+1], advance y_current to the prediction.
    Errors compound naturally.

If the scaffold is structurally sufficient, both modes should track truth.
If structurally insufficient, one-step still fits (local interpolation)
but honest rollout diverges — proving the scaffold cannot represent the
true dynamics regardless of how theta(t) is chosen.

Usage (multi-scaffold):
    python honest_rollout.py \
        --dataset-dir datasets/ \
        --scaffolds reduced2,reduced3,reduced5,reduced7,reduced9,full13 \
        --sample-idx 0 \
        --gd-steps 400 \
        --out results/honest_rollout.png

Usage (single scaffold):
    python honest_rollout.py \
        --dataset datasets/my_data.npz \
        --scaffold reduced5 \
        --sample-idx 0 \
        --out results/honest_rollout.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scaffolds import SCAFFOLDS
from jumps import make_u_to_y_jump

FULL_SPECIES = list("ABCDEFGHIJKLM")
SCAFFOLD_ALIASES = {"reduced13": "full13", "full": "full13"}


def normalize_scaffold_name(name: str) -> str:
    return SCAFFOLD_ALIASES.get(name.strip(), name.strip())


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_npz_obs_names(path: Path):
    try:
        d = np.load(str(path), allow_pickle=True)
        if "obs_names" not in d:
            return None
        return set(d["obs_names"].astype(str).tolist())
    except Exception:
        return None


def find_matching_dataset(scaffold_name: str, dataset_dir: Path):
    sc = SCAFFOLDS[scaffold_name]
    target = set(sc.state_names)
    for p in sorted(dataset_dir.glob("*.npz")):
        if load_npz_obs_names(p) == target:
            return p
    return None


def build_scaffold_dataset_map(scaffold_names, dataset_dir: Path):
    mapping = {}
    for sn in scaffold_names:
        canonical = normalize_scaffold_name(sn)
        if canonical not in SCAFFOLDS:
            print(f"[warn] Unknown scaffold '{sn}' — skipping.")
            continue
        if canonical != sn:
            print(f"[info] Alias '{sn}' -> '{canonical}'.")
        match = find_matching_dataset(canonical, dataset_dir)
        if match is None:
            print(f"[warn] No matching dataset for '{canonical}' "
                  f"(need obs_names={set(SCAFFOLDS[canonical].state_names)}) — skipping.")
            continue
        mapping[canonical] = match
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
#  Numerics
# ─────────────────────────────────────────────────────────────────────────────

def gamma(x, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(x)


def rk4(rhs, y, dt, theta, n_sub=4):
    if dt.ndim == 1:
        dt = dt.unsqueeze(1)
    h = dt / float(n_sub)
    for _ in range(n_sub):
        k1 = rhs(y,                 theta)
        k2 = rhs(y + 0.5 * h * k1, theta)
        k3 = rhs(y + 0.5 * h * k2, theta)
        k4 = rhs(y +       h * k3, theta)
        y  = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return torch.clamp_min(y, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Core: run both one-step oracle and honest rollout
# ─────────────────────────────────────────────────────────────────────────────

def run(scaffold_name, dataset_path, sample_idx,
        gd_steps=400, lr=0.05,
        theta_lo=1e-3, theta_hi=2.0, n_substeps=4,
        device=torch.device("cpu")):
    """
    Returns dict with keys:
        y_full, t_obs, state_names,
        pred_onestep, thetas_onestep, losses_onestep,
        pred_rollout, thetas_rollout, losses_rollout
    """
    sc        = SCAFFOLDS[scaffold_name]
    rhs       = sc.rhs
    theta_dim = sc.theta_dim
    P         = sc.P

    d      = np.load(str(dataset_path), allow_pickle=True)
    y0     = d["y0"][sample_idx].astype(np.float32)
    y_seq  = d["y_seq"][sample_idx].astype(np.float32)
    u_seq  = d["u_seq"][sample_idx].astype(np.float32)
    t_obs  = d["t_obs"].astype(np.float32)
    dt     = np.diff(t_obs).astype(np.float32)

    y_full = np.concatenate([y0[None], y_seq], axis=0)   # (K+1, P)
    K      = len(dt)

    y_true   = torch.from_numpy(y_full).float().to(device)
    u_tensor = torch.from_numpy(u_seq).float().to(device)
    dt_t     = torch.from_numpy(dt).float().to(device)

    jump = make_u_to_y_jump(
        torch.from_numpy(d["control_indices"].astype(np.int64)),
        torch.from_numpy(d["obs_indices"].astype(np.int64)),
        device=device,
    )

    # ── 1) One-step oracle (batched — all K steps in parallel) ────────────────
    y_prev       = y_true[:-1]                       # (K, P)
    y_next       = y_true[1:]                        # (K, P)
    y_after_jump_all = y_prev + (u_tensor @ jump)    # (K, P)

    raw_os = torch.zeros(K, theta_dim, device=device, requires_grad=True)
    opt_os = torch.optim.Adam([raw_os], lr=lr)

    for i in range(gd_steps):
        opt_os.zero_grad()
        theta_b = gamma(raw_os, theta_lo, theta_hi)
        y_hat_b = rk4(rhs, y_after_jump_all, dt_t, theta_b, n_substeps)
        loss = (torch.log1p(y_hat_b) - torch.log1p(y_next)).pow(2).mean()
        loss.backward()
        opt_os.step()

    with torch.no_grad():
        theta_os = gamma(raw_os, theta_lo, theta_hi)
        y_hat_os = rk4(rhs, y_after_jump_all, dt_t, theta_os, n_substeps)
        losses_os = (torch.log1p(y_hat_os) - torch.log1p(y_next)).pow(2).mean(dim=-1).cpu().numpy()

    pred_onestep   = y_hat_os.detach().cpu().numpy()
    thetas_onestep = theta_os.detach().cpu().numpy()
    print(f"  one-step oracle done  (mean loss={float(losses_os.mean()):.5f})")

    # ── 2) Honest rollout (sequential — must be serial) ──────────────────────
    pred_rollout   = np.zeros((K, P))
    thetas_rollout = np.zeros((K, theta_dim))
    losses_rollout = np.zeros(K)

    y_cur = y_true[0].unsqueeze(0).clone()

    for k in range(K):
        u_k      = u_tensor[k].unsqueeze(0)
        dt_k     = dt_t[k].unsqueeze(0)
        y_target = y_true[k + 1].unsqueeze(0)

        y_after_jump_k = y_cur + (u_k @ jump)

        raw = torch.zeros(1, theta_dim, device=device, requires_grad=True)
        opt = torch.optim.Adam([raw], lr=lr)

        for _ in range(gd_steps):
            opt.zero_grad()
            theta_k = gamma(raw, theta_lo, theta_hi)
            y_hat   = rk4(rhs, y_after_jump_k.detach(), dt_k, theta_k, n_substeps)
            loss    = (torch.log1p(y_hat) - torch.log1p(y_target)).pow(2).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            theta_k = gamma(raw, theta_lo, theta_hi)
            y_hat   = rk4(rhs, y_after_jump_k, dt_k, theta_k, n_substeps)
            final_loss = (torch.log1p(y_hat) - torch.log1p(y_target)).pow(2).mean()

        pred_rollout[k]   = y_hat.squeeze(0).detach().cpu().numpy()
        thetas_rollout[k] = theta_k.squeeze(0).detach().cpu().numpy()
        losses_rollout[k] = float(final_loss)

        y_cur = y_hat.detach()

        if k % max(1, K // 10) == 0:
            print(f"  rollout step {k:4d}/{K}  loss={float(final_loss):.5f}")

    return dict(
        y_full         = y_full,
        t_obs          = t_obs,
        state_names    = list(sc.state_names),
        pred_onestep   = pred_onestep,
        thetas_onestep = thetas_onestep,
        losses_onestep = losses_os,
        pred_rollout   = pred_rollout,
        thetas_rollout = thetas_rollout,
        losses_rollout = losses_rollout,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def nrmse(pred, true):
    mask = ~np.isnan(true)
    if mask.sum() == 0:
        return np.nan
    rng = true[mask].max() - true[mask].min()
    if rng < 1e-10:
        return np.nan
    return float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)) / rng)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-scaffold plots (individual)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_scaffold(res, scaffold_name, sample_idx, out_root):
    """Prediction, theta, and loss plots per scaffold."""
    out_dir = out_root / scaffold_name
    out_dir.mkdir(parents=True, exist_ok=True)

    state_names = res["state_names"]
    y_true = res["y_full"][1:]
    tt     = res["t_obs"][1:]
    P      = y_true.shape[1]

    # ── pred vs true ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(P, 1, figsize=(11, max(6, 2.0 * P)), sharex=True)
    if P == 1:
        axes = [axes]

    for p, ax in enumerate(axes):
        ax.plot(tt, y_true[:, p], lw=2, color="tab:blue", label="truth")
        ax.plot(tt, res["pred_onestep"][:, p], lw=1.8, ls="--",
                color="tab:orange", label="one-step")
        ax.plot(tt, res["pred_rollout"][:, p], lw=1.8, ls=":",
                color="tab:red", label="rollout")
        ax.set_ylabel(state_names[p])
        ax.grid(True, alpha=0.25)
        if p == 0:
            ax.legend(fontsize=9)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Oracle per-step fit — {scaffold_name} (sample {sample_idx})")
    fig.tight_layout()
    fig.savefig(out_dir / f"pred_vs_true_{sample_idx:03d}.png", dpi=150)
    plt.close(fig)

    # ── theta (rollout) ──────────────────────────────────────────────────────
    D = res["thetas_rollout"].shape[1]
    n_cols = 2
    n_rows = (D + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.2 * n_rows), sharex=True)
    axes = np.array(axes).reshape(-1)
    for j, ax in enumerate(axes):
        if j < D:
            ax.plot(tt, res["thetas_rollout"][:, j], lw=1.8)
            ax.set_ylabel(f"θ{j}")
            ax.grid(True, alpha=0.25)
        else:
            ax.axis("off")
    axes[min(D - 1, len(axes) - 1)].set_xlabel("Time")
    fig.suptitle(f"Fitted θ(t) [honest rollout] — {scaffold_name} (sample {sample_idx})")
    fig.tight_layout()
    fig.savefig(out_dir / f"theta_sample{sample_idx}.png", dpi=150)
    plt.close(fig)

    # ── GD loss ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.semilogy(tt, res["losses_onestep"], lw=1.5, label="one-step", color="tab:orange")
    ax.semilogy(tt, res["losses_rollout"], lw=1.5, label="rollout",  color="tab:red")
    ax.set_xlabel("Time")
    ax.set_ylabel("GD loss (log)")
    ax.set_title(f"Per-step GD loss at convergence — {scaffold_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "gd_losses.png", dpi=150)
    plt.close(fig)

    print(f"  [{scaffold_name}] per-scaffold plots -> {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-scaffold summary plot (A/M grid)
# ─────────────────────────────────────────────────────────────────────────────

def make_summary_plot(all_results, scaffold_order, show_species,
                      sample_idx, out_path):
    """
    Grid plot: rows = show_species, columns = scaffolds.
    Each panel shows truth, one-step, and rollout — matching the style of
    the per-scaffold pred_vs_true plots.
    """
    n_scaffolds = len(scaffold_order)
    n_species   = len(show_species)

    # scale: ~3 inches per column, ~2.5 per row, but cap column width
    col_w = max(2.2, min(3.5, 18.0 / n_scaffolds))
    row_h = 2.8
    fig, axes = plt.subplots(
        n_species, n_scaffolds,
        figsize=(col_w * n_scaffolds + 0.8, row_h * n_species + 1.2),
        squeeze=False, sharex=False,
    )

    for col, sn in enumerate(scaffold_order):
        res         = all_results[sn]
        state_names = res["state_names"]
        y_true      = res["y_full"][1:]
        tt          = res["t_obs"][1:]

        for row, sp in enumerate(show_species):
            ax = axes[row][col]

            if sp not in state_names:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"{sn}\n(P={SCAFFOLDS[sn].P})", fontsize=9)
                continue

            sc_idx = state_names.index(sp)
            gt = y_true[:, sc_idx]
            os = res["pred_onestep"][:, sc_idx]
            ro = res["pred_rollout"][:, sc_idx]

            ax.plot(tt, gt, lw=2,   color="tab:blue",   label="truth")
            ax.plot(tt, os, lw=1.8, color="tab:orange",  ls="--", label="one-step")
            ax.plot(tt, ro, lw=1.8, color="tab:red",     ls=":",  label="rollout")

            # NRMSE annotations
            n_os = nrmse(os, gt)
            n_ro = nrmse(ro, gt)
            ax.text(0.97, 0.95,
                    f"OS={n_os:.3f}\nRO={n_ro:.3f}",
                    transform=ax.transAxes, fontsize=7,
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

            ax.grid(True, alpha=0.2)

            if row == 0:
                ax.set_title(f"{sn}\n(P={SCAFFOLDS[sn].P})", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"species {sp}", fontsize=10)
            if row == n_species - 1:
                ax.set_xlabel("time", fontsize=8)

            # legend only top-left panel
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        f"Oracle per-step theta fitting  (sample {sample_idx})\n"
        f"Each scaffold uses its own matched dataset  |  "
        f"one-step: starts from y_true[k]  |  rollout: honest (from own prediction)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary plot     -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  NRMSE bar chart
# ─────────────────────────────────────────────────────────────────────────────

def make_nrmse_plot(all_results, scaffold_order, show_species, out_path):
    n_sp = len(show_species)
    fig, axes = plt.subplots(1, n_sp, figsize=(5 * n_sp, 4), squeeze=False)
    x     = np.arange(len(scaffold_order))
    bar_w = 0.35

    for s_i, sp in enumerate(show_species):
        ax = axes[0][s_i]
        nrmse_os, nrmse_ro, sizes = [], [], []

        for sn in scaffold_order:
            res = all_results[sn]
            state_names = res["state_names"]
            sizes.append(SCAFFOLDS[sn].P)

            if sp in state_names:
                sc_idx = state_names.index(sp)
                gt = res["y_full"][1:, sc_idx]
                nrmse_os.append(nrmse(res["pred_onestep"][:, sc_idx], gt))
                nrmse_ro.append(nrmse(res["pred_rollout"][:, sc_idx], gt))
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
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"NRMSE plot       -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CSV export
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(all_results, scaffold_order, sample_idx, export_dir):
    """
    Per scaffold, save:
      predictions_sample{idx}.csv  — time, true, one-step, rollout per species
      theta_rollout_sample{idx}.csv
      theta_onestep_sample{idx}.csv
      losses_sample{idx}.csv
    """
    export_dir.mkdir(parents=True, exist_ok=True)

    for sn in scaffold_order:
        res = all_results[sn]
        state_names = res["state_names"]
        tt = res["t_obs"][1:]
        y_true = res["y_full"][1:]

        sc_dir = export_dir / sn
        sc_dir.mkdir(parents=True, exist_ok=True)

        # predictions
        cols = [tt[:, None], y_true, res["pred_onestep"], res["pred_rollout"]]
        header = (
            ["time"]
            + [f"true_{s}" for s in state_names]
            + [f"onestep_{s}" for s in state_names]
            + [f"rollout_{s}" for s in state_names]
        )
        np.savetxt(
            sc_dir / f"predictions_sample{sample_idx}.csv",
            np.concatenate(cols, axis=1),
            delimiter=",", header=",".join(header), comments="",
        )

        # theta (rollout)
        D = res["thetas_rollout"].shape[1]
        theta_header = ["time"] + [f"theta_{j}" for j in range(D)]
        np.savetxt(
            sc_dir / f"theta_rollout_sample{sample_idx}.csv",
            np.concatenate([tt[:, None], res["thetas_rollout"]], axis=1),
            delimiter=",", header=",".join(theta_header), comments="",
        )

        # theta (one-step)
        np.savetxt(
            sc_dir / f"theta_onestep_sample{sample_idx}.csv",
            np.concatenate([tt[:, None], res["thetas_onestep"]], axis=1),
            delimiter=",", header=",".join(theta_header), comments="",
        )

        # losses
        loss_header = ["time", "loss_onestep", "loss_rollout"]
        np.savetxt(
            sc_dir / f"losses_sample{sample_idx}.csv",
            np.column_stack([tt, res["losses_onestep"], res["losses_rollout"]]),
            delimiter=",", header=",".join(loss_header), comments="",
        )

    print(f"CSV exports      -> {export_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Oracle per-step theta: one-step + honest rollout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset",     default=None,
                        help="Single .npz path (single-scaffold mode).")
    parser.add_argument("--scaffold",    default="reduced5",
                        help="Scaffold for single-scaffold mode.")
    parser.add_argument("--dataset-dir", default=None,
                        help="Directory of .npz files for auto-matching.")
    parser.add_argument("--scaffolds",
                        default="reduced2,reduced3,reduced5,reduced7,reduced9,full13",
                        help="Comma-separated scaffold names.")
    parser.add_argument("--all-scaffolds", action="store_true")
    parser.add_argument("--sample-idx",  type=int,   default=0)
    parser.add_argument("--show-species", type=str,  default="A,M")
    parser.add_argument("--gd-steps",    type=int,   default=400)
    parser.add_argument("--lr",          type=float, default=0.05)
    parser.add_argument("--n-substeps",  type=int,   default=4)
    parser.add_argument("--theta-lo",    type=float, default=1e-3)
    parser.add_argument("--theta-hi",    type=float, default=2.0)
    parser.add_argument("--out",         default="results/honest_rollout.png")
    args = parser.parse_args()

    device = device_auto()
    print(f"Device: {device}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_root = out_path.parent / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    # ── resolve scaffolds & datasets ──────────────────────────────────────────
    if args.dataset is not None:
        sn = normalize_scaffold_name(args.scaffold)
        if sn not in SCAFFOLDS:
            print(f"Unknown scaffold '{args.scaffold}'. "
                  f"Available: {list(SCAFFOLDS.keys())}")
            sys.exit(1)
        run_items = [(sn, Path(args.dataset))]
    else:
        if args.dataset_dir is None:
            print("[error] Provide --dataset (single) or --dataset-dir (multi).")
            sys.exit(1)

        if args.all_scaffolds:
            scaffold_names = list(SCAFFOLDS.keys())
        else:
            raw = [s.strip() for s in args.scaffolds.split(",") if s.strip()]
            scaffold_names = list(dict.fromkeys(
                normalize_scaffold_name(s) for s in raw
            ))

        sd_map = build_scaffold_dataset_map(scaffold_names, Path(args.dataset_dir))
        valid = [sn for sn in scaffold_names if sn in sd_map]

        if not valid:
            print("[error] No scaffolds matched to a dataset. Aborting.")
            sys.exit(1)

        print(f"\n{'Scaffold':<30}  {'P':>3}  Dataset")
        print("-" * 80)
        for sn in valid:
            print(f"  {sn:<28}  {SCAFFOLDS[sn].P:>3d}  {sd_map[sn].name}")
        print()
        run_items = [(sn, sd_map[sn]) for sn in valid]

    show_species = [s.strip().upper() for s in args.show_species.split(",")
                    if s.strip().upper() in FULL_SPECIES]

    # ── run fitting ───────────────────────────────────────────────────────────
    all_results = {}
    scaffold_order = []

    for scaffold_name, dataset_path in run_items:
        sc = SCAFFOLDS[scaffold_name]
        print(f"\n>> {scaffold_name}  (P={sc.P}, θ_dim={sc.theta_dim})")
        print(f"   dataset: {dataset_path}")

        res = run(
            scaffold_name, str(dataset_path), args.sample_idx,
            gd_steps   = args.gd_steps,
            lr         = args.lr,
            theta_lo   = args.theta_lo,
            theta_hi   = args.theta_hi,
            n_substeps = args.n_substeps,
            device     = device,
        )
        all_results[scaffold_name] = res
        scaffold_order.append(scaffold_name)

        # per-scaffold plots
        plot_per_scaffold(res, scaffold_name, args.sample_idx, plot_root)

    # ── NRMSE table ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    header = f"{'Scaffold':<28} {'P':>3}"
    for sp in show_species:
        header += f"  OS-{sp:>1}    RO-{sp:>1}"
    print(header)
    print("=" * 80)

    for sn in scaffold_order:
        res = all_results[sn]
        state_names = res["state_names"]
        row = f"{sn:<28} {SCAFFOLDS[sn].P:>3d}"
        for sp in show_species:
            if sp in state_names:
                sc_idx = state_names.index(sp)
                gt = res["y_full"][1:, sc_idx]
                row += f"  {nrmse(res['pred_onestep'][:, sc_idx], gt):.4f}"
                row += f"  {nrmse(res['pred_rollout'][:, sc_idx], gt):.4f}"
            else:
                row += "    N/A     N/A"
        print(row)

    print("=" * 80)
    print("OS = one-step (from y_true[k])  |  RO = honest rollout (from own prediction)\n")

    # ── summary outputs ───────────────────────────────────────────────────────
    if len(scaffold_order) > 1:
        make_summary_plot(all_results, scaffold_order, show_species,
                          args.sample_idx, out_path)

        nrmse_path = out_path.parent / (out_path.stem + "_nrmse.png")
        make_nrmse_plot(all_results, scaffold_order, show_species, nrmse_path)

    export_dir = out_path.parent / "exports"
    export_csv(all_results, scaffold_order, args.sample_idx, export_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()