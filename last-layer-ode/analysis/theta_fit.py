"""
theta_fit.py

Unified oracle (manual) per-step theta fit + plotting.

Two subcommands:

    fit   Batched GD theta fit on one, many, or all samples of a dataset for a
          chosen scaffold. Writes results.npz + nrmse.csv to --out.

    plot  Load a previously-saved results.npz and produce per-sample figures.
          No fitting.

Sample selection (--samples) accepts:
    all              every sample in the dataset
    7                first 7 samples (0..6)
    0,3,17           explicit list
    0-99             inclusive range

Examples:
    # fit all samples, plot them too
    python analysis/theta_fit.py fit \\
        --dataset datasets/real_ivtt_raw_no_outliers.npz \\
        --scaffold txtl_resource_and_maturation_dna \\
        --loss-species mm,pm --samples all \\
        --out results/txtl_fit --plot

    # fit a single sample (no rollout, fast)
    python analysis/theta_fit.py fit --dataset D.npz --scaffold s5 \\
        --samples 0 --no-rollout --out results/quick

    # replot a subset of samples from an existing fit
    python analysis/theta_fit.py plot --results results/txtl_fit \\
        --samples 0,3,7 --fmt png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scaffolds import SCAFFOLDS
from jumps import make_u_to_y_jump


# ── helpers ────────────────────────────────────────────────────────────────────

def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x))


def rk4_batch(rhs, y, dt, theta, n_sub=4):
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


def nrmse_np(pred, true):
    mask = ~np.isnan(true)
    if mask.sum() == 0:
        return float("nan")
    rng = float(true[mask].max() - true[mask].min())
    if rng < 1e-10:
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)) / rng)


def device_auto():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_samples(spec: str, total_N: int) -> list[int]:
    """all | <N> | i,j,k | i-j"""
    spec = spec.strip()
    if spec == "all" or spec == "-1":
        return list(range(total_N))
    if "," in spec:
        idxs = [int(s) for s in spec.split(",") if s.strip()]
    elif "-" in spec:
        a, b = spec.split("-")
        idxs = list(range(int(a), int(b) + 1))
    else:
        # Single integer: 0 or 1 means "that index"; >1 means "first n samples".
        n = int(spec)
        idxs = [n] if n <= 1 else list(range(min(n, total_N)))
    bad = [i for i in idxs if i < 0 or i >= total_N]
    if bad:
        raise ValueError(f"Sample indices out of range [0,{total_N}): {bad}")
    return idxs


def parse_csv_list(s: str | None) -> list[str] | None:
    if not s:
        return None
    return [tok.strip() for tok in s.split(",") if tok.strip()]


# ── core fit ───────────────────────────────────────────────────────────────────

def build_pm_time_weights(
    K: int,
    mode: str,
    endpoint_weight: float,
    tail_weight: float,
    device: torch.device,
) -> torch.Tensor:
    """Per-timestep multiplier on the pm residual, shape (K,).

    mode='none'   : ones everywhere (default theta_fit behavior).
    mode='simple' : ones, except w[-1] = endpoint_weight. Just nudges the final
                    pm residual harder.
    mode='composite' : weighted pm scheme matching train.py's composite loss —
                      • second half of trajectory (k >= K//2) weighted by tail_weight
                      • last step gets an additional endpoint_weight on top.
                    Defaults (tail=3, endpoint=0.1) match the composite loss_weight scheme.
                    ('bob' is a backward-compatible alias for 'composite'.)
    """
    w = torch.ones(K, device=device)
    if mode == "none":
        return w
    if mode == "simple":
        w[-1] = float(endpoint_weight)
        return w
    if mode in ("composite", "bob"):
        if K > 1:
            w[K // 2:] = float(tail_weight)
        w[-1] = float(tail_weight) + float(endpoint_weight)
        return w
    raise ValueError(f"Unknown pm_endpoint_mode: {mode!r}")


def fit(
    scaffold_name: str,
    dataset_path: str,
    sample_indices: list[int],
    gd_steps: int = 400,
    lr: float = 0.05,
    n_substeps: int = 4,
    loss_species: list[str] | None = None,
    no_rollout: bool = False,
    pm_endpoint_mode: str = "none",
    pm_endpoint_weight: float = 5.0,
    pm_tail_weight: float = 3.0,
    device: torch.device = torch.device("cpu"),
) -> dict:
    sc = SCAFFOLDS[scaffold_name]
    theta_dim, P = sc.theta_dim, sc.P
    rhs = sc
    is_analytic = bool(getattr(sc, "has_analytic_step", False))

    d = np.load(dataset_path, allow_pickle=True)
    t_obs = d["t_obs"].astype(np.float32)
    dt_np = np.diff(t_obs).astype(np.float32)
    K = len(dt_np)

    N = len(sample_indices)
    y0_np = d["y0"][sample_indices].astype(np.float32)
    y_np  = d["y_seq"][sample_indices].astype(np.float32)
    u_np  = d["u_seq"][sample_indices].astype(np.float32)
    y_true_np = np.concatenate([y0_np[:, None, :], y_np], axis=1)  # (N, K+1, P)

    jump = make_u_to_y_jump(
        torch.from_numpy(d["control_indices"].astype(np.int64)),
        torch.from_numpy(d["obs_indices"].astype(np.int64)),
        device=device,
    )

    y_true = torch.from_numpy(y_true_np).to(device)
    u_seq  = torch.from_numpy(u_np).to(device)
    dt_t   = torch.from_numpy(dt_np).to(device)

    if sc.theta_lo_vec is not None and sc.theta_hi_vec is not None:
        lo_t = torch.tensor(sc.theta_lo_vec, dtype=torch.float32, device=device)
        hi_t = torch.tensor(sc.theta_hi_vec, dtype=torch.float32, device=device)
    else:
        lo_t = torch.full((theta_dim,), 1e-3, device=device)
        hi_t = torch.full((theta_dim,), 2.0,  device=device)

    state_names = list(sc.state_names)
    if loss_species:
        missing = [s for s in loss_species if s not in state_names]
        if missing:
            raise ValueError(f"loss_species {missing} not in scaffold {state_names}")
        loss_idx = torch.tensor([state_names.index(s) for s in loss_species],
                                dtype=torch.long, device=device)
        loss_species_local = list(loss_species)
    else:
        loss_idx = torch.arange(P, device=device)
        loss_species_local = list(state_names)

    # ── per-timestep species weights (shape (K, |loss_species|)) ──────────────
    # Only pm gets a non-uniform time profile; everything else is 1.
    P_loss = len(loss_species_local)
    species_weight = torch.ones(K, P_loss, device=device)
    pm_loss_pos = loss_species_local.index("pm") if "pm" in loss_species_local else None
    if pm_loss_pos is not None and pm_endpoint_mode != "none":
        pm_w = build_pm_time_weights(K, pm_endpoint_mode, pm_endpoint_weight,
                                     pm_tail_weight, device)
        species_weight[:, pm_loss_pos] = pm_w
        print(f"  pm-endpoint loss: mode={pm_endpoint_mode}  "
              f"endpoint_w={pm_endpoint_weight}  tail_w={pm_tail_weight}  "
              f"pm_w[K-1]={float(pm_w[-1]):.3f}  pm_w[K//2]={float(pm_w[K//2]):.3f}")
    sw_sum = species_weight.sum()  # scalar normalizer for one-step

    # Analytic-scaffold precompute (e.g. IVTT's dna_cum_total). Empty ctx for non-analytic.
    if is_analytic:
        ctx = sc.precompute_batch(y_true[:, 0, :], u_seq)
    else:
        ctx = {}

    def step_fn(y, dt, theta):
        """Single integration step. Dispatches on scaffold capability."""
        if is_analytic:
            return sc.analytic_step(y, dt, theta, ctx)
        return rk4_batch(rhs, y, dt, theta, n_substeps)

    # ── one-step oracle: all N×K timesteps fit in parallel ─────────────────────
    # Analytic scaffolds carry hidden internal state (e.g. IVTT's R, O) absent from
    # the dataset truth — "step from y_true[k]" is ill-defined for them, so skip.
    if is_analytic:
        print("  [analytic scaffold] skipping one-step oracle (truth lacks hidden states).")
        pred_os   = np.zeros((N, K, P), dtype=np.float32)
        thetas_os = np.zeros((N, K, theta_dim), dtype=np.float32)
    else:
        y_prev = y_true[:, :-1, :]
        y_next = y_true[:, 1:,  :]
        y_jmp  = y_prev + (u_seq @ jump)
        NK = N * K
        y_flat  = y_jmp.reshape(NK, P)
        dt_flat = dt_t.unsqueeze(0).expand(N, -1).reshape(NK)

        raw_os = torch.zeros(N, K, theta_dim, device=device, requires_grad=True)
        opt_os = torch.optim.Adam([raw_os], lr=lr)
        print(f"  One-step oracle: N={N}, K={K}, theta_dim={theta_dim}  [{gd_steps} steps]")
        for step in range(gd_steps):
            opt_os.zero_grad()
            theta_flat = log_gamma(raw_os.reshape(NK, theta_dim), lo_t, hi_t)
            y_hat = step_fn(y_flat, dt_flat, theta_flat).reshape(N, K, P)
            diff  = torch.log1p(y_hat) - torch.log1p(y_next)
            diff_sel = diff.index_select(-1, loss_idx)        # (N, K, P_loss)
            # weighted MSE: sum(w * diff^2) / (N * sum(w))
            loss  = (diff_sel.pow(2) * species_weight).sum() / (N * sw_sum)
            loss.backward()
            opt_os.step()
            if (step + 1) % 100 == 0:
                print(f"    step {step+1:>4}/{gd_steps}  loss={loss.item():.6f}")

        with torch.no_grad():
            theta_final = log_gamma(raw_os.reshape(NK, theta_dim), lo_t, hi_t)
            pred_os = step_fn(y_flat, dt_flat, theta_final).reshape(N, K, P).cpu().numpy()
            thetas_os = log_gamma(raw_os, lo_t, hi_t).cpu().numpy()

    # ── honest rollout: N samples in parallel, K timesteps sequential ──────────
    if no_rollout:
        pred_ro   = np.zeros((N, K, P), dtype=np.float32)
        thetas_ro = np.zeros((N, K, theta_dim), dtype=np.float32)
    else:
        pred_ro_t   = torch.zeros(N, K, P,         device=device)
        thetas_ro_t = torch.zeros(N, K, theta_dim, device=device)
        # Analytic scaffolds may reseed hidden states (e.g. IVTT sets R=O=1, m=p=0.01).
        y_cur = sc.initial_state(y_true[:, 0, :]) if is_analytic else y_true[:, 0, :].clone()
        print(f"  Rollout ({'analytic' if is_analytic else 'rk4'}): "
              f"{K} timesteps × {gd_steps} steps each")
        for k in range(K):
            u_k   = u_seq[:, k, :]
            dt_k  = dt_t[k].expand(N)
            y_tgt = y_true[:, k + 1, :]
            y_in  = y_cur + (u_k @ jump)

            raw_k = torch.zeros(N, theta_dim, device=device, requires_grad=True)
            opt_k = torch.optim.Adam([raw_k], lr=lr)
            w_k = species_weight[k]                       # (P_loss,)
            w_k_sum = w_k.sum()
            for _ in range(gd_steps):
                opt_k.zero_grad()
                theta_k = log_gamma(raw_k, lo_t, hi_t)
                y_hat_k = step_fn(y_in.detach(), dt_k, theta_k)
                diff_k  = torch.log1p(y_hat_k) - torch.log1p(y_tgt)
                diff_k_sel = diff_k.index_select(-1, loss_idx)   # (N, P_loss)
                loss_k = (diff_k_sel.pow(2) * w_k).sum() / (N * w_k_sum)
                loss_k.backward()
                opt_k.step()

            with torch.no_grad():
                theta_k = log_gamma(raw_k, lo_t, hi_t)
                y_hat_k = step_fn(y_in, dt_k, theta_k)
            pred_ro_t[:, k, :]   = y_hat_k
            thetas_ro_t[:, k, :] = theta_k
            y_cur = y_hat_k.detach()
            if (k + 1) % max(1, K // 10) == 0:
                print(f"    rollout step {k+1:>4}/{K}")

        pred_ro   = pred_ro_t.cpu().numpy()
        thetas_ro = thetas_ro_t.cpu().numpy()

    return dict(
        scaffold_name  = scaffold_name,
        y_true         = y_true_np,
        t_obs          = t_obs,
        state_names    = state_names,
        sample_indices = list(sample_indices),
        pred_onestep   = pred_os,
        thetas_onestep = thetas_os,
        pred_rollout   = pred_ro,
        thetas_rollout = thetas_ro,
        no_rollout     = bool(no_rollout),
    )


# ── saving / loading ───────────────────────────────────────────────────────────

def save_results(results: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.npz"
    np.savez_compressed(
        path,
        scaffold_name  = np.array(results["scaffold_name"]),
        y_true         = results["y_true"],
        t_obs          = results["t_obs"],
        state_names    = np.array(results["state_names"]),
        sample_indices = np.array(results["sample_indices"]),
        pred_onestep   = results["pred_onestep"],
        thetas_onestep = results["thetas_onestep"],
        pred_rollout   = results["pred_rollout"],
        thetas_rollout = results["thetas_rollout"],
        no_rollout     = np.array(results["no_rollout"]),
    )
    print(f"Saved results -> {path}")
    return path


def load_results(path: Path) -> dict:
    if path.is_dir():
        path = path / "results.npz"
    z = np.load(path, allow_pickle=True)
    return dict(
        scaffold_name  = str(z["scaffold_name"]),
        y_true         = z["y_true"],
        t_obs          = z["t_obs"],
        state_names    = [str(s) for s in z["state_names"]],
        sample_indices = [int(i) for i in z["sample_indices"]],
        pred_onestep   = z["pred_onestep"],
        thetas_onestep = z["thetas_onestep"],
        pred_rollout   = z["pred_rollout"],
        thetas_rollout = z["thetas_rollout"],
        no_rollout     = bool(z["no_rollout"]),
    )


def save_nrmse_csv(results: dict, out_dir: Path,
                   eval_species: list[str] | None = None) -> None:
    state_names = results["state_names"]
    species = eval_species if eval_species else state_names
    y_true  = results["y_true"][:, 1:, :]
    pred_os = results["pred_onestep"]
    pred_ro = results["pred_rollout"]
    sample_idxs = results["sample_indices"]
    no_ro = results["no_rollout"]

    rows = []
    agg = {"os": {sp: [] for sp in species}, "ro": {sp: [] for sp in species}}
    for ni, si in enumerate(sample_idxs):
        row = {"sample_idx": si}
        for sp in species:
            if sp not in state_names:
                continue
            j = state_names.index(sp)
            n_os = nrmse_np(pred_os[ni, :, j], y_true[ni, :, j])
            row[f"os_{sp}"] = f"{n_os:.6f}"
            agg["os"][sp].append(n_os)
            if not no_ro:
                n_ro = nrmse_np(pred_ro[ni, :, j], y_true[ni, :, j])
                row[f"ro_{sp}"] = f"{n_ro:.6f}"
                agg["ro"][sp].append(n_ro)
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "nrmse.csv"
    metrics = ("os",) if no_ro else ("os", "ro")
    fieldnames = ["sample_idx"] + [f"{m}_{sp}" for sp in species for m in metrics]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        summary = {"sample_idx": "MEAN"}
        for sp in species:
            for m in metrics:
                vals = [v for v in agg[m].get(sp, []) if not np.isnan(v)]
                summary[f"{m}_{sp}"] = f"{float(np.mean(vals)):.6f}" if vals else ""
        w.writerow(summary)
    print(f"Saved NRMSE   -> {csv_path}")

    header = f"\n{'Species':<8}  {'OS mean':>10}"
    if not no_ro:
        header += f"  {'RO mean':>10}"
    print(header)
    print("-" * len(header))
    for sp in species:
        os_vals = [v for v in agg["os"].get(sp, []) if not np.isnan(v)]
        os_mean = float(np.mean(os_vals)) if os_vals else float("nan")
        line = f"  {sp:<6}  {os_mean:>10.4f}"
        if not no_ro:
            ro_vals = [v for v in agg["ro"].get(sp, []) if not np.isnan(v)]
            ro_mean = float(np.mean(ro_vals)) if ro_vals else float("nan")
            line += f"  {ro_mean:>10.4f}"
        print(line)


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_sample(ni: int, si: int, results: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state_names = results["state_names"]
    tt = results["t_obs"][1:]
    P = len(state_names)
    no_ro = results["no_rollout"]

    y_true   = results["y_true"][ni, 1:, :]
    pred_os  = results["pred_onestep"][ni]
    pred_ro  = results["pred_rollout"][ni]
    theta_os = results["thetas_onestep"][ni]
    theta_ro = results["thetas_rollout"][ni]
    theta_dim = theta_os.shape[1]

    n_theta_rows = (theta_dim + 1) // 2
    n_rows = max(P, n_theta_rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13.5, n_rows * 2.2), squeeze=False)

    for i, sp in enumerate(state_names):
        ax = axes[i][0]
        ax.plot(tt, y_true[:, i], lw=2.0, color="tab:blue", label="truth")
        ax.plot(tt, pred_os[:, i], lw=1.5, color="tab:green", ls="--",
                label=f"one-step  NRMSE={nrmse_np(pred_os[:, i], y_true[:, i]):.3f}")
        if not no_ro:
            ax.plot(tt, pred_ro[:, i], lw=1.5, color="tab:red", ls=":",
                    label=f"rollout   NRMSE={nrmse_np(pred_ro[:, i], y_true[:, i]):.3f}")
        ax.set_ylabel(sp, fontsize=11)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
    for i in range(P, n_rows):
        axes[i][0].axis("off")

    for j in range(theta_dim):
        row, col = j // 2, 1 + (j % 2)
        ax = axes[row][col]
        ax.plot(tt, theta_os[:, j], lw=1.2, color="tab:purple", label="OS")
        if not no_ro:
            ax.plot(tt, theta_ro[:, j], lw=1.2, color="tab:orange", ls="--", label="RO")
        ax.set_ylabel(f"θ{j}", fontsize=9)
        ax.grid(True, alpha=0.2)
        if j == 0:
            ax.legend(fontsize=7)
    for j in range(theta_dim, n_rows * 2):
        axes[j // 2][1 + (j % 2)].axis("off")

    axes[-1][0].set_xlabel("Time")
    axes[-1][1].set_xlabel("Time")
    axes[-1][2].set_xlabel("Time")
    axes[0][0].set_title("Predictions", fontsize=10)
    axes[0][1].set_title("θ one-step", fontsize=10)
    axes[0][2].set_title("θ rollout",  fontsize=10)
    fig.suptitle(f"Oracle theta fit — {results['scaffold_name']} — sample {si}", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_results(results: dict, out_dir: Path, which: list[int] | None, fmt: str) -> None:
    sample_idxs = results["sample_indices"]
    if which is None:
        targets = list(range(len(sample_idxs)))
    else:
        idx_map = {si: ni for ni, si in enumerate(sample_idxs)}
        missing = [s for s in which if s not in idx_map]
        if missing:
            raise ValueError(f"Samples {missing} not in results (have {sample_idxs}).")
        targets = [idx_map[s] for s in which]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Plotting {len(targets)} samples -> {out_dir}")
    for ni in targets:
        si = sample_idxs[ni]
        plot_sample(ni, si, results, out_dir / f"sample_{si:04d}.{fmt}")
    print(f"  Done.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def add_common_sample_arg(p):
    p.add_argument("--samples", default="all",
                   help="Sample selector: 'all' | <N> first N | 'i,j,k' | 'i-j'.")


def cmd_fit(args):
    if args.scaffold not in SCAFFOLDS:
        print(f"Unknown scaffold '{args.scaffold}'. Available: {list(SCAFFOLDS.keys())}")
        sys.exit(1)

    d = np.load(args.dataset, allow_pickle=True)
    total_N = d["y0"].shape[0]
    sample_indices = parse_samples(args.samples, total_N)

    loss_species = parse_csv_list(args.loss_species)
    eval_species = parse_csv_list(args.eval_species) or loss_species

    device = device_auto()
    print(f"Device: {device}")
    print(f"Scaffold: {args.scaffold}  |  N={len(sample_indices)} of {total_N}  "
          f"|  gd_steps={args.gd_steps}  |  loss={loss_species}")

    results = fit(
        scaffold_name      = args.scaffold,
        dataset_path       = args.dataset,
        sample_indices     = sample_indices,
        gd_steps           = args.gd_steps,
        lr                 = args.lr,
        n_substeps         = args.n_substeps,
        loss_species       = loss_species,
        no_rollout         = args.no_rollout,
        pm_endpoint_mode   = args.pm_endpoint_mode,
        pm_endpoint_weight = args.pm_endpoint_weight,
        pm_tail_weight     = args.pm_tail_weight,
        device             = device,
    )

    out_dir = Path(args.out)
    save_results(results, out_dir)
    save_nrmse_csv(results, out_dir, eval_species)

    if args.plot:
        plot_results(results, out_dir / "plots", which=None, fmt=args.fmt)

    print("\nDone.")


def cmd_plot(args):
    results = load_results(Path(args.results))
    which = None
    if args.samples != "all":
        which = parse_samples(args.samples, max(results["sample_indices"]) + 1)
    out_dir = Path(args.out) if args.out else Path(args.results) / "plots"
    plot_results(results, out_dir, which=which, fmt=args.fmt)
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Unified oracle theta fit + plot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fit", help="Run GD theta fit on one/many/all samples.")
    pf.add_argument("--dataset", required=True)
    pf.add_argument("--scaffold", required=True)
    add_common_sample_arg(pf)
    pf.add_argument("--gd-steps",     type=int,   default=400)
    pf.add_argument("--lr",           type=float, default=0.05)
    pf.add_argument("--n-substeps",   type=int,   default=4)
    pf.add_argument("--loss-species", type=str, default=None,
                    help="Comma list of species in the loss (default: all).")
    pf.add_argument("--eval-species", type=str, default=None,
                    help="Comma list for NRMSE table (default: loss-species).")
    pf.add_argument("--no-rollout",   action="store_true",
                    help="Skip honest rollout (one-step oracle only).")
    pf.add_argument("--pm-endpoint-mode", choices=["none", "simple", "composite", "bob"],
                    default="none",
                    help="Add extra weight on pm in the loss. "
                         "'simple' weights only pm at the last step by "
                         "--pm-endpoint-weight. 'composite' matches the train.py "
                         "weighted loss: second half of trajectory gets --pm-tail-weight "
                         "on pm, plus an extra --pm-endpoint-weight at the final step. "
                         "('bob' is a backward-compatible alias for 'composite'.)")
    pf.add_argument("--pm-endpoint-weight", type=float, default=5.0,
                    help="Endpoint pm weight. Default 5.0 for 'simple'; "
                         "use 0.1 for 'composite'.")
    pf.add_argument("--pm-tail-weight", type=float, default=3.0,
                    help="Second-half pm weight in 'composite' mode (default 3.0).")
    pf.add_argument("--out",          required=True, help="Output directory.")
    pf.add_argument("--plot",         action="store_true",
                    help="Also plot every fitted sample after saving.")
    pf.add_argument("--fmt",          default="pdf", help="Plot format (pdf|png).")
    pf.set_defaults(func=cmd_fit)

    pp = sub.add_parser("plot", help="Replot from a saved results.npz.")
    pp.add_argument("--results", required=True,
                    help="Path to results.npz or its containing dir.")
    add_common_sample_arg(pp)
    pp.add_argument("--out", default=None,
                    help="Output dir for figures (default: <results>/plots).")
    pp.add_argument("--fmt", default="pdf", help="Plot format (pdf|png).")
    pp.set_defaults(func=cmd_plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
