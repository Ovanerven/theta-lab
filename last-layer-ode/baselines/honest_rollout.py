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

sys.path.insert(0, str(Path(__file__).parent.parent))
from scaffolds import SCAFFOLDS
from jumps import make_u_to_y_jump

FULL_SPECIES = list("ABCDEFGHIJKLM")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available.")
        return torch.device("cuda")
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


def build_scaffold_dataset_map(scaffold_names, dataset_dir, explicit_pairs):
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
        device: torch.device = torch.device("cpu")):

    sc        = SCAFFOLDS[scaffold_name]
    sc = sc.to(device)
    rhs       = sc
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
    preds_t      = torch.zeros((K, P), device=device)
    thetas_out_t = torch.zeros((K, theta_dim), device=device)
    losses_out_t = torch.zeros(K, device=device)

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

        preds_t[k]      = y_hat.squeeze(0).detach()
        thetas_out_t[k] = theta_k.squeeze(0).detach()
        losses_out_t[k] = final_loss.detach()

        # ← key difference: advance from OUR prediction, not y_true
        y_cur = y_hat.detach()

        if k % max(1, K // 10) == 0:
            print(f"  step {k:4d}/{K}  loss={float(final_loss.detach().cpu()):.5f}")

    preds = preds_t.detach().cpu().numpy()
    thetas_out = thetas_out_t.detach().cpu().numpy()
    losses_out = losses_out_t.detach().cpu().numpy()
    return y_full, preds, thetas_out, losses_out, t_obs, list(sc.state_names)


# ─────────────────────────────────────────────────────────────────────────────

def plot_results(y_full, preds, thetas, losses, t_obs, state_names,
                 scaffold_name, sample_idx, out_path):
    P = preds.shape[1]
    t = t_obs[1:]

    # predictions
    fig, axes = plt.subplots(P, 1, figsize=(11, max(6, 2.0 * P)), sharex=True)
    if P == 1:
        axes = [axes]
    for p, ax in enumerate(axes):
        ax.plot(t, y_full[1:, p], linewidth=2, label="true")
        ax.plot(t, preds[:, p], linewidth=2, linestyle="--", label="pred")
        ax.set_ylabel(state_names[p] if p < len(state_names) else f"s{p}")
        ax.grid(alpha=0.25)
        if p == 0:
            ax.legend()
    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Prediction vs truth (sample {sample_idx})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Prediction plot -> {out_path}")

    # GD loss over time
    loss_path = Path(out_path).parent / (Path(out_path).stem + "_losses.png")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.semilogy(t, losses, lw=1.5)
    ax.set_xlabel("Time"); ax.set_ylabel("GD loss (log)")
    ax.set_title(f"Per-step GD loss at convergence — {scaffold_name}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"Loss plot       -> {loss_path}")

    # theta over time
    theta_path = Path(out_path).parent / (Path(out_path).stem + "_theta.png")
    D = thetas.shape[1]
    n_cols = 2
    n_rows = (D + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.0 * n_rows), sharex=True)
    axes = np.array(axes).reshape(-1)
    for j in range(len(axes)):
        if j < D:
            axes[j].plot(t, thetas[:, j], lw=1.5)
            axes[j].set_ylabel(f"θ{j}")
            axes[j].grid(alpha=0.25)
        else:
            axes[j].axis("off")
    axes[-1].set_xlabel("Time")
    fig.suptitle(f"θ(t) — {scaffold_name}")
    fig.tight_layout()
    fig.savefig(theta_path, dpi=150)
    plt.close(fig)
    print(f"Theta plot      -> {theta_path}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default=None,
                        help="Single-dataset mode: explicit dataset path")
    parser.add_argument("--dataset-dir", default=None,
                        help="Multi-scaffold mode: auto-match datasets by obs_names")
    parser.add_argument("--scaffold-datasets", default=None,
                        help="Optional explicit pairs: 'scaffold:path,...'")
    parser.add_argument("--scaffold",   default="reduced5",
                        help="Single-scaffold mode")
    parser.add_argument("--scaffolds",  default=None,
                        help="Comma-separated scaffold list for multi-scaffold mode")
    parser.add_argument("--sample-idx", type=int,   default=0)
    parser.add_argument("--gd-steps",   type=int,   default=400)
    parser.add_argument("--lr",         type=float, default=0.05)
    parser.add_argument("--n-substeps", type=int,   default=4)
    parser.add_argument("--theta-lo",   type=float, default=1e-3)
    parser.add_argument("--theta-hi",   type=float, default=2.0)
    parser.add_argument("--device",     choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--show-species", default=None,
                        help="Accepted for CLI compatibility; not used in this script.")
    parser.add_argument("--out",        default="results/honest_rollout.png")
    args = parser.parse_args()

    explicit_pairs = {}
    if args.scaffold_datasets:
        for token in args.scaffold_datasets.split(","):
            token = token.strip()
            if ":" not in token:
                print(f"[warn] Ignoring malformed token: '{token}'")
                continue
            sn, path_str = token.split(":", 1)
            explicit_pairs[sn.strip()] = path_str.strip()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_base = Path(args.out)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    if args.scaffolds is not None:
        scaffold_names = [s.strip() for s in args.scaffolds.split(",") if s.strip()]
        dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
        sd_map = build_scaffold_dataset_map(scaffold_names, dataset_dir, explicit_pairs)
        valid_scaffolds = [sn for sn in scaffold_names if sn in sd_map]

        if not valid_scaffolds:
            print("[error] No scaffolds matched to a dataset. Aborting.")
            sys.exit(1)

        for sn in valid_scaffolds:
            out_path = out_base.parent / f"{out_base.stem}_{sn}{out_base.suffix}"
            print(f"\nScaffold: {sn} | sample: {args.sample_idx} | gd_steps: {args.gd_steps}")
            print(f"Dataset: {sd_map[sn]}")

            y_full, preds, thetas, losses, t_obs, state_names = run(
                scaffold_name=sn,
                dataset_path=str(sd_map[sn]),
                sample_idx=args.sample_idx,
                gd_steps=args.gd_steps,
                lr=args.lr,
                theta_lo=args.theta_lo,
                theta_hi=args.theta_hi,
                n_substeps=args.n_substeps,
                device=device,
            )

            plot_results(y_full, preds, thetas, losses, t_obs, state_names,
                         sn, args.sample_idx, str(out_path))

            print("NRMSE per species:")
            for p, name in enumerate(state_names):
                gt = y_full[1:, p]
                rng = gt.max() - gt.min()
                if rng > 1e-10:
                    err = float(np.sqrt(np.mean((preds[:, p] - gt) ** 2)) / rng)
                    print(f"  {name}: {err:.4f}")
    else:
        if args.scaffold not in SCAFFOLDS:
            print(f"Unknown scaffold '{args.scaffold}'. Available: {list(SCAFFOLDS.keys())}")
            sys.exit(1)
        if args.dataset is None:
            print("[error] --dataset is required in single-scaffold mode.")
            sys.exit(1)

        print(f"Scaffold: {args.scaffold}  |  sample: {args.sample_idx}  "
              f"|  gd_steps: {args.gd_steps}")

        y_full, preds, thetas, losses, t_obs, state_names = run(
            scaffold_name=args.scaffold,
            dataset_path=args.dataset,
            sample_idx=args.sample_idx,
            gd_steps=args.gd_steps,
            lr=args.lr,
            theta_lo=args.theta_lo,
            theta_hi=args.theta_hi,
            n_substeps=args.n_substeps,
            device=device,
        )

        plot_results(y_full, preds, thetas, losses, t_obs, state_names,
                     args.scaffold, args.sample_idx, args.out)

        print("\nNRMSE per species:")
        for p, name in enumerate(state_names):
            gt = y_full[1:, p]
            rng = gt.max() - gt.min()
            if rng > 1e-10:
                err = float(np.sqrt(np.mean((preds[:, p] - gt) ** 2)) / rng)
                print(f"  {name}: {err:.4f}")


if __name__ == "__main__":
    main()