#!/usr/bin/env python3
"""
fit_theta_per_step.py

Oracle per-step theta fitting experiment across mechanistic scaffolds.

For each scaffold, the *correct* matching dataset is loaded automatically —
the one whose observed (and controlled) species exactly equal the scaffold's
state names.  This matters because in each dataset:

    control_indices == obs_indices == scaffold state indices

so every bolus hits an observed species and the experiment is self-consistent.
Using the full-13 dataset for a smaller scaffold would mean boluses on
unobserved species get silently ignored, distorting the comparison.

At each discrete time step k the optimisation problem solved is:

    theta_k* = argmin_theta  loss( RK4(y_true[k], theta, dt_k), y_true[k+1] )

This is the *oracle* per-step theta: the best the scaffold could ever achieve
given the true previous state.  Residual loss after GD convergence therefore
measures structural mismatch of the ODE scaffold, not learning capacity.

Two prediction modes are computed:
  one-step  -- starts from y_true[k] at every step
  rollout   -- chains predictions (errors compound)

Usage
-----
# Auto-discover all matching datasets in a directory:
python fit_theta_per_step.py \\
    --dataset-dir  datasets/ \\
    --scaffolds    reduced2,reduced3,reduced5,reduced7,reduced9,full13 \\
    --sample-idx   0 \\
    --gd-steps     400 \\
    --show-species A,M \\
    --out          results/theta_per_step.png

# Or supply explicit scaffold->dataset pairs (overrides auto-discovery):
python fit_theta_per_step.py \\
    --scaffold-datasets "reduced5:datasets/N1000_T300_steps600_zeros_knoise0.0_reduced5_ADGJM.npz,full13:datasets/N1000_T300_steps600_zeros_knoise0.0_full13_ABCDEFGHIJKLM.npz" \\
    --sample-idx   0 \\
    --gd-steps     400 \\
    --out          results/theta_per_step.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── make project modules importable ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))  # last-layer-ode/
from scaffolds import SCAFFOLDS, Scaffold
from jumps import make_u_to_y_jump

# ── constants ─────────────────────────────────────────────────────────────────
FULL_SPECIES = list("ABCDEFGHIJKLM")


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset auto-matching
# ─────────────────────────────────────────────────────────────────────────────

def load_npz_obs_names(path: Path):
    """Return the set of obs_names strings from a .npz, or None on failure."""
    try:
        d = np.load(str(path), allow_pickle=True)
        if "obs_names" not in d:
            return None
        return set(d["obs_names"].astype(str).tolist())
    except Exception:
        return None


def find_matching_dataset(scaffold_name: str, dataset_dir: Path):
    """
    Scan dataset_dir for the .npz whose obs_names exactly match the scaffold's
    state_names.  Returns the first match path, or None.
    """
    sc     = SCAFFOLDS[scaffold_name]
    target = set(sc.state_names)

    for p in sorted(dataset_dir.glob("*.npz")):
        obs = load_npz_obs_names(p)
        if obs == target:
            return p
    return None


def build_scaffold_dataset_map(scaffold_names, dataset_dir, explicit_pairs):
    """
    Returns {scaffold_name: dataset_path} for every scaffold we can match.
    explicit_pairs (dict scaffold->path-string) takes priority.
    """
    mapping = {}
    for sn in scaffold_names:
        if sn not in SCAFFOLDS:
            print(f"[warn] Unknown scaffold '{sn}' — skipping.")
            continue

        if sn in explicit_pairs:
            p = Path(explicit_pairs[sn])
            if not p.exists():
                print(f"[warn] Explicit dataset for '{sn}' not found: {p} — skipping.")
                continue
            mapping[sn] = p
            continue

        if dataset_dir is None:
            print(f"[warn] No --dataset-dir and no explicit dataset for '{sn}' — skipping.")
            continue

        match = find_matching_dataset(sn, dataset_dir)
        if match is None:
            print(f"[warn] No matching dataset for '{sn}' "
                  f"(need obs_names={set(SCAFFOLDS[sn].state_names)}) — skipping.")
            continue
        mapping[sn] = match

    return mapping


# ─────────────────────────────────────────────────────────────────────────────
#  Load one sample from a matched dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_sample(npz_path: Path, sample_idx: int) -> dict:
    """
    Returns dict:
      y_full          (K+1, P)  full trajectory including y0
      u_seq           (K,   U)
      dt              (K,)
      control_indices (U,)      full-state indices
      obs_indices     (P,)      full-state indices
      obs_names       list[str]
      t_obs           (K+1,)
    """
    d   = np.load(str(npz_path), allow_pickle=True)
    idx = sample_idx

    y0    = d["y0"][idx].astype(np.float32)
    y_seq = d["y_seq"][idx].astype(np.float32)
    u_seq = d["u_seq"][idx].astype(np.float32)
    t_obs = d["t_obs"].astype(np.float32)

    y_full = np.concatenate([y0[None], y_seq], axis=0)
    dt     = np.diff(t_obs).astype(np.float32)

    control_indices = d["control_indices"].astype(np.int64)
    obs_indices     = d["obs_indices"].astype(np.int64)
    obs_names       = d["obs_names"].astype(str).tolist()

    return dict(
        y_full=y_full,
        u_seq=u_seq,
        dt=dt,
        control_indices=control_indices,
        obs_indices=obs_indices,
        obs_names=obs_names,
        t_obs=t_obs,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Numerics helpers
# ─────────────────────────────────────────────────────────────────────────────

def gamma(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


def rk4(rhs, y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor,
         n_sub: int = 4) -> torch.Tensor:
    """RK4 with n_sub substeps.  y:(B,P)  dt:(B,)  theta:(B,D)."""
    if dt.ndim == 1:
        dt = dt.unsqueeze(1)
    h = dt / float(n_sub)
    for _ in range(n_sub):
        k1 = rhs(y,                 theta)
        k2 = rhs(y + 0.5 * h * k1, theta)
        k3 = rhs(y + 0.5 * h * k2, theta)
        k4 = rhs(y + h       * k3, theta)
        y  = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return torch.clamp_min(y, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-step oracle fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_scaffold_per_step(
    scaffold_name: str,
    sample: dict,
    gd_steps: int   = 400,
    lr: float       = 0.05,
    theta_lo: float = 1e-3,
    theta_hi: float = 2.0,
    n_substeps: int = 4,
    device          = torch.device("cpu"),
    verbose: bool   = False,
) -> dict:
    """
    Oracle per-step theta fitting for one scaffold using its matched sample.

    Because the sample comes from the scaffold's own dataset:
      - control_indices == obs_indices == scaffold state indices
      - every bolus channel is an observed species
      - the jump matrix is well-defined and complete (no ignored boluses)

    Returns dict with pred_onestep, pred_rollout, fit_losses, thetas,
    state_names, obs_idx, t_obs.
    """
    sc          = SCAFFOLDS[scaffold_name]
    rhs         = sc.rhs
    theta_dim   = sc.theta_dim
    state_names = sc.state_names
    P           = sc.P

    obs_idx   = np.array([FULL_SPECIES.index(n) for n in state_names], dtype=np.int64)
    y_sc      = torch.from_numpy(sample["y_full"]).float().to(device)   # (K+1, P)
    u_tensor  = torch.from_numpy(sample["u_seq"]).float().to(device)    # (K,   U)
    dt_tensor = torch.from_numpy(sample["dt"]).float().to(device)       # (K,)
    K         = int(dt_tensor.shape[0])

    # Jump: maps bolus channel -> scaffold state.
    # control_indices == obs_indices for matched datasets, so every bolus
    # channel corresponds to an observed scaffold species.
    jump = make_u_to_y_jump(
        torch.from_numpy(sample["control_indices"]).long(),
        torch.from_numpy(sample["obs_indices"]).long(),
        device=device,
    )   # (U, P)

    pred_onestep = torch.zeros(K, P)
    fit_losses   = np.zeros(K)
    thetas       = torch.zeros(K, theta_dim)

    for k in range(K):
        y_prev = y_sc[k].unsqueeze(0)          # (1, P)  true start
        y_next = y_sc[k + 1].unsqueeze(0)      # (1, P)  target
        u_k    = u_tensor[k].unsqueeze(0)       # (1, U)
        dt_k   = dt_tensor[k].unsqueeze(0)      # (1,)

        y_after_jump = y_prev + (u_k @ jump)   # apply bolus before integration

        # Fresh raw theta each step — no warm start — clean oracle
        raw = torch.zeros(1, theta_dim, device=device, requires_grad=True)
        opt = torch.optim.Adam([raw], lr=lr)

        for _ in range(gd_steps):
            opt.zero_grad()
            theta_k = gamma(raw, theta_lo, theta_hi)
            y_hat   = rk4(rhs, y_after_jump, dt_k, theta_k, n_substeps)
            loss    = (torch.log1p(y_hat) - torch.log1p(y_next)).pow(2).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            theta_k    = gamma(raw, theta_lo, theta_hi)
            y_hat      = rk4(rhs, y_after_jump, dt_k, theta_k, n_substeps)
            final_loss = (torch.log1p(y_hat) - torch.log1p(y_next)).pow(2).mean()

        pred_onestep[k] = y_hat.squeeze(0).cpu()
        fit_losses[k]   = float(final_loss)
        thetas[k]       = theta_k.squeeze(0).detach().cpu()

        if verbose and k % max(1, K // 10) == 0:
            print(f"    t-step {k:4d}/{K}  GD loss = {float(final_loss):.6f}")

    # Rollout using the fitted thetas
    pred_rollout = torch.zeros(K, P)
    y_cur = y_sc[0].unsqueeze(0).clone().to(device)

    for k in range(K):
        u_k     = u_tensor[k].unsqueeze(0)
        dt_k    = dt_tensor[k].unsqueeze(0)
        theta_k = thetas[k].unsqueeze(0).to(device)

        y_after_jump = y_cur + (u_k @ jump)
        y_cur        = rk4(rhs, y_after_jump, dt_k, theta_k, n_substeps)
        pred_rollout[k] = y_cur.squeeze(0).cpu()

    return dict(
        pred_onestep = pred_onestep.numpy(),
        pred_rollout = pred_rollout.numpy(),
        fit_losses   = fit_losses,
        thetas       = thetas.numpy(),
        state_names  = state_names,
        obs_idx      = obs_idx,
        t_obs        = sample["t_obs"],
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def nrmse(pred: np.ndarray, true: np.ndarray) -> float:
    mask = ~np.isnan(true)
    if mask.sum() == 0:
        return np.nan
    rng = true[mask].max() - true[mask].min()
    if rng < 1e-10:
        return np.nan
    return float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)) / rng)


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────

def _species_panel(ax, t_obs, y_full, pred_os, pred_ro, sp, scaffold_name,
                   show_legend=False):
    ax.plot(t_obs,      y_full,  color="tab:blue",   lw=1.6, ls="-",  label="truth",    alpha=0.85)
    ax.plot(t_obs[1:],  pred_os, color="tab:orange", lw=1.4, ls="--", label="one-step", alpha=0.90)
    ax.plot(t_obs[1:],  pred_ro, color="tab:red",    lw=1.2, ls=":",  label="rollout",  alpha=0.90)
    ax.set_title(f"{scaffold_name}  |  species {sp}", fontsize=8.5, pad=3)
    ax.set_xlabel("time", fontsize=7)
    ax.set_ylabel("conc.", fontsize=7)
    ax.tick_params(labelsize=6)
    if show_legend:
        ax.legend(fontsize=6, loc="upper right")


def make_trajectory_plot(results, scaffold_names, show_species, sample_idx, out_path):
    n_sc = len(scaffold_names)
    n_sp = len(show_species)
    fig, axes = plt.subplots(n_sp, n_sc, figsize=(3.5 * n_sc, 3.0 * n_sp), squeeze=False)

    for col, sn in enumerate(scaffold_names):
        res         = results[sn]
        state_names = list(res["state_names"])
        t_obs       = res["t_obs"]
        y_full      = res["y_full"]

        for row, sp in enumerate(show_species):
            ax = axes[row][col]
            if sp in state_names:
                sc_idx = state_names.index(sp)
                _species_panel(
                    ax, t_obs,
                    y_full[:, sc_idx],
                    res["pred_onestep"][:, sc_idx],
                    res["pred_rollout"][:, sc_idx],
                    sp, sn,
                    show_legend=(col == 0 and row == 0),
                )
            else:
                ax.text(0.5, 0.5, f"species {sp}\nnot in scaffold",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="grey")
                ax.set_title(f"{sn}  |  species {sp}", fontsize=8.5, pad=3)

    plt.suptitle(
        f"Oracle per-step theta fitting  (sample {sample_idx})\n"
        "Each scaffold uses its own matched dataset  |  "
        "one-step: starts from y_true[k]  |  rollout: chained",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Trajectory plot  -> {out_path}")


def make_nrmse_plot(results, scaffold_names, show_species, out_path):
    n_sp   = len(show_species)
    fig, axes = plt.subplots(1, n_sp, figsize=(5 * n_sp, 4), squeeze=False)
    x      = np.arange(len(scaffold_names))
    bar_w  = 0.35

    for s_i, sp in enumerate(show_species):
        ax = axes[0][s_i]
        nrmse_os, nrmse_ro, sizes = [], [], []

        for sn in scaffold_names:
            res         = results[sn]
            state_names = list(res["state_names"])
            sizes.append(SCAFFOLDS[sn].P)

            if sp in state_names:
                sc_idx = state_names.index(sp)
                gt     = res["y_full"][1:, sc_idx]
                nrmse_os.append(nrmse(res["pred_onestep"][:, sc_idx], gt))
                nrmse_ro.append(nrmse(res["pred_rollout"][:, sc_idx], gt))
            else:
                nrmse_os.append(np.nan)
                nrmse_ro.append(np.nan)

        ax.bar(x - bar_w / 2, nrmse_os, bar_w, label="one-step",
               color="tab:orange", alpha=0.85)
        ax.bar(x + bar_w / 2, nrmse_ro, bar_w, label="rollout",
               color="tab:red",    alpha=0.85)
        ax.set_title(f"NRMSE — species {sp}", fontsize=10)
        ax.set_xlabel("scaffold")
        ax.set_ylabel("NRMSE")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{sn}\n(P={s})" for sn, s in zip(scaffold_names, sizes)],
            fontsize=7, rotation=30, ha="right",
        )
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Oracle per-step theta — NRMSE\n(each scaffold uses its own matched dataset)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    p = out_path.parent / (out_path.stem + "_nrmse.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"NRMSE plot       -> {p}")


def make_loss_plot(results, scaffold_names, out_path):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    colors  = plt.cm.tab10(np.linspace(0, 0.9, len(scaffold_names)))

    for sn, col in zip(scaffold_names, colors):
        losses = results[sn]["fit_losses"]
        t_pred = results[sn]["t_obs"][1: len(losses) + 1]
        ax.semilogy(t_pred, losses,
                    label=f"{sn} (P={SCAFFOLDS[sn].P})",
                    lw=1.5, color=col)

    ax.set_xlabel("time")
    ax.set_ylabel("GD loss at convergence (log scale)")
    ax.set_title(
        "Residual per-step GD loss — irreducible scaffold mismatch error\n"
        "Higher floor = scaffold structurally cannot fit the one-step transition"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = out_path.parent / (out_path.stem + "_gd_losses.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"GD-loss plot     -> {p}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Oracle per-step theta fitting across mechanistic scaffolds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir", type=str, default=None,
        help="Directory to scan for .npz files.  Each scaffold is auto-matched "
             "to the file whose obs_names exactly equal the scaffold state_names.",
    )
    parser.add_argument(
        "--scaffold-datasets", type=str, default=None,
        help="Explicit overrides: 'scaffold:path,...' pairs, e.g. "
             "'reduced5:datasets/foo.npz,full13:datasets/bar.npz'. "
             "Takes priority over --dataset-dir for listed scaffolds.",
    )
    parser.add_argument(
        "--scaffolds", type=str,
        default="reduced2,reduced3,reduced5,reduced7,reduced9,full13",
        help="Comma-separated scaffold names (display order).",
    )
    parser.add_argument("--sample-idx",    type=int,   default=0)
    parser.add_argument("--show-species",  type=str,   default="A,M",
                        help="Species to plot, e.g. 'A,M'.")
    parser.add_argument("--gd-steps",      type=int,   default=400,
                        help="Adam iterations per time step.")
    parser.add_argument("--lr",            type=float, default=0.05)
    parser.add_argument("--n-substeps",    type=int,   default=4,
                        help="RK4 substeps per interval.")
    parser.add_argument("--theta-lo",      type=float, default=1e-3)
    parser.add_argument("--theta-hi",      type=float, default=2.0)
    parser.add_argument("--out",           type=str,   default="results/theta_per_step.png")
    parser.add_argument("--verbose",       action="store_true")
    args = parser.parse_args()

    scaffold_names = [s.strip() for s in args.scaffolds.split(",") if s.strip()]

    explicit_pairs = {}
    if args.scaffold_datasets:
        for token in args.scaffold_datasets.split(","):
            token = token.strip()
            if ":" not in token:
                print(f"[warn] Ignoring malformed token: '{token}'")
                continue
            sn, path_str = token.split(":", 1)
            explicit_pairs[sn.strip()] = path_str.strip()

    dataset_dir   = Path(args.dataset_dir) if args.dataset_dir else None
    sd_map        = build_scaffold_dataset_map(scaffold_names, dataset_dir, explicit_pairs)
    valid_scaffolds = [sn for sn in scaffold_names if sn in sd_map]

    if not valid_scaffolds:
        print("[error] No scaffolds matched to a dataset. Aborting.")
        sys.exit(1)

    print(f"\n{'Scaffold':<30}  {'P':>3}  Dataset")
    print("-" * 80)
    for sn in valid_scaffolds:
        print(f"  {sn:<28}  {SCAFFOLDS[sn].P:>3d}  {sd_map[sn].name}")
    print()

    show_species = [s.strip().upper() for s in args.show_species.split(",")
                    if s.strip().upper() in FULL_SPECIES]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  GD steps: {args.gd_steps}  |  lr: {args.lr}  "
          f"|  sample idx: {args.sample_idx}\n")

    # ── fit ───────────────────────────────────────────────────────────────────
    results = {}
    for sn in valid_scaffolds:
        sc = SCAFFOLDS[sn]
        print(f">> {sn}  (P={sc.P}, theta_dim={sc.theta_dim})")
        print(f"   dataset: {sd_map[sn]}")

        sample = load_sample(sd_map[sn], args.sample_idx)
        print(f"   K={sample['dt'].shape[0]}, U={sample['u_seq'].shape[1]}, "
              f"obs={sample['obs_names']}")

        res = fit_scaffold_per_step(
            sn, sample,
            gd_steps   = args.gd_steps,
            lr         = args.lr,
            theta_lo   = args.theta_lo,
            theta_hi   = args.theta_hi,
            n_substeps = args.n_substeps,
            device     = device,
            verbose    = args.verbose,
        )
        res["y_full"] = sample["y_full"]      # attach for plotting

        mean_loss = float(res["fit_losses"].mean())
        max_loss  = float(res["fit_losses"].max())
        print(f"   GD loss: mean={mean_loss:.5f}  max={max_loss:.5f}\n")
        results[sn] = res

    # ── NRMSE table ───────────────────────────────────────────────────────────
    print("=" * 72)
    header = f"{'Scaffold':<28} {'P':>3}"
    for sp in show_species:
        header += f"  OS-{sp}   RO-{sp}"
    print(header)
    print("=" * 72)

    for sn in valid_scaffolds:
        res         = results[sn]
        state_names = list(res["state_names"])
        row         = f"{sn:<28} {SCAFFOLDS[sn].P:>3d}"
        for sp in show_species:
            if sp in state_names:
                sc_idx = state_names.index(sp)
                gt     = res["y_full"][1:, sc_idx]
                n_os   = nrmse(res["pred_onestep"][:, sc_idx], gt)
                n_ro   = nrmse(res["pred_rollout"][:, sc_idx], gt)
                row   += f"  {n_os:.4f}  {n_ro:.4f}"
            else:
                row += "    N/A     N/A"
        print(row)

    print("=" * 72)
    print("OS = one-step (from y_true[k])  |  RO = rollout (chained from y0)")

    # ── plots ─────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    make_trajectory_plot(results, valid_scaffolds, show_species, args.sample_idx, out_path)
    make_nrmse_plot     (results, valid_scaffolds, show_species, out_path)
    make_loss_plot      (results, valid_scaffolds, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()