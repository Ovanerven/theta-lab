"""
honest_rollout.py

At each step k:
  1. Start from y_current  (your own previous prediction, not ground truth)
  2. Optimise theta to minimise loss(rk4(y_current, theta, dt_k), y_true[k+1])
  3. Apply that theta to get y_current for the next step

This is the honest version: errors compound naturally.

Usage:
    python honest_rollout.py \
        --dataset  datasets/my_data.npz \
        --scaffold reduced5 \
        --sample-idx 0 \
        --gd-steps 400 \
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
SCAFFOLD_ALIASES = {
    "reduced13": "full13",
    "full": "full13",
}


def normalize_scaffold_name(name: str) -> str:
    n = name.strip()
    return SCAFFOLD_ALIASES.get(n, n)


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        obs = load_npz_obs_names(p)
        if obs == target:
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
            print(f"[info] Using scaffold alias '{sn}' -> '{canonical}'.")

        match = find_matching_dataset(canonical, dataset_dir)
        if match is None:
            print(f"[warn] No matching dataset for '{canonical}' "
                  f"(need obs_names={set(SCAFFOLDS[canonical].state_names)}) — skipping.")
            continue
        mapping[canonical] = match
    return mapping


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

def run(scaffold_name, dataset_path, sample_idx,
        gd_steps=400, lr=0.05,
    theta_lo=1e-3, theta_hi=2.0, n_substeps=4,
    device=torch.device("cpu")):

    sc        = SCAFFOLDS[scaffold_name]
    rhs       = sc.rhs
    theta_dim = sc.theta_dim
    P         = sc.P

    # load data
    d      = np.load(dataset_path, allow_pickle=True)
    y0     = d["y0"][sample_idx].astype(np.float32)
    y_seq  = d["y_seq"][sample_idx].astype(np.float32)
    u_seq  = d["u_seq"][sample_idx].astype(np.float32)
    t_obs  = d["t_obs"].astype(np.float32)
    dt     = np.diff(t_obs).astype(np.float32)

    y_full = np.concatenate([y0[None], y_seq], axis=0)   # (K+1, P)
    K      = len(dt)

    y_true   = torch.from_numpy(y_full).float().to(device)           # (K+1, P)
    u_tensor = torch.from_numpy(u_seq).float().to(device)            # (K,   U)
    dt_t     = torch.from_numpy(dt).float().to(device)               # (K,)

    jump = make_u_to_y_jump(
        torch.from_numpy(d["control_indices"].astype(np.int64)),
        torch.from_numpy(d["obs_indices"].astype(np.int64)),
        device=device,
    )   # (U, P)

    # ── honest rollout ────────────────────────────────────────────────────────
    preds      = np.zeros((K, P))
    thetas_out = np.zeros((K, theta_dim))
    losses_out = np.zeros(K)

    y_cur = y_true[0].unsqueeze(0).clone()   # start from true y0

    for k in range(K):
        u_k  = u_tensor[k].unsqueeze(0)
        dt_k = dt_t[k].unsqueeze(0)
        y_target = y_true[k + 1].unsqueeze(0)

        # apply bolus to CURRENT state (not ground truth)
        y_after_jump = y_cur + (u_k @ jump)

        # optimise theta from where we actually are
        raw = torch.zeros(1, theta_dim, device=device, requires_grad=True)
        opt = torch.optim.Adam([raw], lr=lr)

        for _ in range(gd_steps):
            opt.zero_grad()
            theta_k = gamma(raw, theta_lo, theta_hi)
            y_hat   = rk4(rhs, y_after_jump.detach(), dt_k, theta_k, n_substeps)
            loss    = (torch.log1p(y_hat) - torch.log1p(y_target)).pow(2).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            theta_k = gamma(raw, theta_lo, theta_hi)
            y_hat   = rk4(rhs, y_after_jump, dt_k, theta_k, n_substeps)
            final_loss = (torch.log1p(y_hat) - torch.log1p(y_target)).pow(2).mean()

        preds[k]      = y_hat.squeeze(0).detach().cpu().numpy()
        thetas_out[k] = theta_k.squeeze(0).detach().cpu().numpy()
        losses_out[k] = float(final_loss)

        # ← key difference: advance from OUR prediction, not y_true
        y_cur = y_hat.detach()

        if k % max(1, K // 10) == 0:
            print(f"  step {k:4d}/{K}  loss={float(final_loss):.5f}")

    return y_full, preds, thetas_out, losses_out, t_obs, list(sc.state_names)


# ─────────────────────────────────────────────────────────────────────────────

def plot_results(y_full, preds, thetas, losses, t_obs, state_names,
                 scaffold_name, sample_idx, out_path):
    out_root = Path(out_path).parent / "plots"
    out_dir = out_root / scaffold_name
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = y_full[1:]  # (K, P)
    t = t_obs
    tt = t[1:]
    P = preds.shape[1]

    # Prediction plot (style matched to plot_diagnostics.plot_predictions)
    pred_path = out_dir / "pred_vs_true_000.png"
    fig, axes = plt.subplots(P, 1, figsize=(11, max(6, 2.0 * P)), sharex=True)
    if P == 1:
        axes = [axes]

    for p, ax in enumerate(axes):
        ax.plot(tt, y_true[:, p], linewidth=2, label="true")
        ax.plot(tt, preds[:, p], linewidth=2, linestyle="--", label="pred")
        ax.set_ylabel(state_names[p] if p < len(state_names) else f"s{p}")
        ax.grid(True, alpha=0.25)
        if p == 0:
            ax.legend()

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Prediction vs truth [rollout] (sample {sample_idx})")
    fig.tight_layout()
    fig.savefig(pred_path, dpi=150)
    plt.close(fig)
    print(f"Prediction plot -> {pred_path}")

    # Theta plot (style matched to plot_diagnostics.plot_theta)
    theta_path = out_dir / f"theta_sample{sample_idx}.png"
    _, D = thetas.shape
    n_cols = 2
    n_rows = (D + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.2 * n_rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for j, ax in enumerate(axes):
        if j < D:
            ax.plot(tt, thetas[:, j], linewidth=1.8)
            ax.set_ylabel(f"θ{j}")
            ax.grid(True, alpha=0.25)
        else:
            ax.axis("off")

    axes[min(D - 1, len(axes) - 1)].set_xlabel("Time")
    fig.suptitle(f"Learned θ(t) (sample {sample_idx})")
    fig.tight_layout()
    fig.savefig(theta_path, dpi=150)
    plt.close(fig)
    print(f"Theta plot      -> {theta_path}")

    # Keep loss diagnostic as an extra
    loss_path = out_dir / "gd_losses.png"
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.semilogy(tt, losses, lw=1.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("GD loss (log)")
    ax.set_title(f"Per-step GD loss at convergence — {scaffold_name}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"Loss plot       -> {loss_path}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default=None,
                        help="Single dataset .npz path (single-scaffold mode).")
    parser.add_argument("--scaffold",   default="reduced5",
                        help="Scaffold for single-scaffold mode.")
    parser.add_argument("--dataset-dir", default=None,
                        help="Directory containing .npz files for auto-matching.")
    parser.add_argument("--scaffolds", default="reduced2,reduced3,reduced5,reduced7,reduced9,full13",
                        help="Comma-separated scaffold names for dataset-dir mode.")
    parser.add_argument("--all-scaffolds", action="store_true",
                        help="Use all scaffolds from SCAFFOLDS in dataset-dir mode.")
    parser.add_argument("--sample-idx", type=int,   default=0)
    parser.add_argument("--gd-steps",   type=int,   default=400)
    parser.add_argument("--lr",         type=float, default=0.05)
    parser.add_argument("--n-substeps", type=int,   default=4)
    parser.add_argument("--theta-lo",   type=float, default=1e-3)
    parser.add_argument("--theta-hi",   type=float, default=2.0)
    parser.add_argument("--out",        default="results/honest_rollout.png")
    args = parser.parse_args()

    device = device_auto()
    print(f"Device: {device}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    run_items = []
    if args.dataset is not None:
        scaffold_name = normalize_scaffold_name(args.scaffold)
        if scaffold_name not in SCAFFOLDS:
            print(f"Unknown scaffold '{args.scaffold}'. Available: {list(SCAFFOLDS.keys())}")
            sys.exit(1)
        run_items = [(scaffold_name, Path(args.dataset))]
    else:
        if args.dataset_dir is None:
            print("[error] Provide either --dataset (single mode) or --dataset-dir (multi mode).")
            sys.exit(1)

        if args.all_scaffolds:
            scaffold_names = list(SCAFFOLDS.keys())
        else:
            raw_scaffold_names = [s.strip() for s in args.scaffolds.split(",") if s.strip()]
            scaffold_names = [normalize_scaffold_name(s) for s in raw_scaffold_names]
            seen = set()
            scaffold_names = [s for s in scaffold_names if not (s in seen or seen.add(s))]

        dataset_dir = Path(args.dataset_dir)
        sd_map = build_scaffold_dataset_map(scaffold_names, dataset_dir)
        valid_scaffolds = [sn for sn in scaffold_names if sn in sd_map]

        if not valid_scaffolds:
            print("[error] No scaffolds matched to a dataset. Aborting.")
            sys.exit(1)

        print(f"\n{'Scaffold':<30}  {'P':>3}  Dataset")
        print("-" * 80)
        for sn in valid_scaffolds:
            print(f"  {sn:<28}  {SCAFFOLDS[sn].P:>3d}  {sd_map[sn].name}")
        print()

        run_items = [(sn, sd_map[sn]) for sn in valid_scaffolds]

    for scaffold_name, dataset_path in run_items:
        print(f"Scaffold: {scaffold_name}  |  sample: {args.sample_idx}  "
              f"|  gd_steps: {args.gd_steps}")

        y_full, preds, thetas, losses, t_obs, state_names = run(
            scaffold_name = scaffold_name,
            dataset_path  = dataset_path,
            sample_idx    = args.sample_idx,
            gd_steps      = args.gd_steps,
            lr            = args.lr,
            theta_lo      = args.theta_lo,
            theta_hi      = args.theta_hi,
            n_substeps    = args.n_substeps,
            device        = device,
        )

        plot_results(y_full, preds, thetas, losses, t_obs, state_names,
                     scaffold_name, args.sample_idx, args.out)

        print("\nNRMSE per species:")
        for p, name in enumerate(state_names):
            gt  = y_full[1:, p]
            rng = gt.max() - gt.min()
            if rng > 1e-10:
                err = float(np.sqrt(np.mean((preds[:, p] - gt)**2) / (rng**2)))
                print(f"  {name}: {err:.4f}")
        print()


if __name__ == "__main__":
    main()