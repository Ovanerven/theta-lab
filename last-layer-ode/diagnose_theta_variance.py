"""
Run inference on the test split of a saved run and plot per-sample theta
trajectories. Answers: is theta essentially constant across samples (mean
collapse), or does it vary experiment-to-experiment?

Usage:
    python last-layer-ode/diagnose_theta_variance.py \
        experiments/txtl_obs_norm_sweep/20260428_115706_mse_minmax
"""
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from plot_diagnostics import rebuild_model_from_experiment, device_auto, load_yaml

@torch.no_grad()
def main():
    exp_dir = Path(sys.argv[1])
    cfg = load_yaml(exp_dir / "config.yaml")
    device = device_auto()

    model, ds, _state_names, _param_names = rebuild_model_from_experiment(exp_dir, device=device)

    split_path = exp_dir / "split.npz"
    if split_path.exists():
        split = np.load(split_path)
        test_idx = split["test_idx"].tolist()
    else:
        test_idx = list(range(len(ds)))

    P_obs = ds.y_seq.shape[-1]
    obs_idx = torch.tensor(cfg.get("obs_idx", []), device=device) if cfg.get("obs_idx") else torch.arange(P_obs, device=device)
    dt_tensor = torch.from_numpy(ds.dt).to(device)

    all_theta = []   # list of (K_i, theta_dim)
    all_pm_final = []

    for i in test_idx:
        y0 = torch.from_numpy(ds.y_seq[i, 0:1]).float().to(device)     # (1, P)
        if ds.variable_length:
            L = int(ds.lengths[i])
            u = torch.from_numpy(ds.u_seq[i, :L]).float().unsqueeze(0).to(device)
            y = torch.from_numpy(ds.y_seq[i, :L]).float().unsqueeze(0).to(device)
        else:
            u = torch.from_numpy(ds.u_seq[i]).float().unsqueeze(0).to(device)
            y = torch.from_numpy(ds.y_seq[i]).float().unsqueeze(0).to(device)
        K = u.shape[1]
        dt = dt_tensor[:K].unsqueeze(0)

        _, theta, _ = model(
            y0, u, dt, obs_idx, y,
            teacher_forcing=False,
            tf_every=int(cfg.get("tf_every", 50)),
            u_transform=str(cfg.get("u_transform", "none")),
            y_transform=str(cfg.get("y_transform", "none")),
        )
        all_theta.append(theta.squeeze(0).cpu().numpy())      # (K, theta_dim)
        all_pm_final.append(float(ds.y_seq[i, -1, 5]))        # pm = col 5

    max_k = max(th.shape[0] for th in all_theta)
    d = all_theta[0].shape[1]
    padded = np.full((len(all_theta), max_k, d), np.nan, dtype=np.float32)
    for i, th in enumerate(all_theta):
        padded[i, : th.shape[0], :] = th
    all_theta = padded   # (N, K_max, D) with NaNs for padding
    all_pm_final = np.array(all_pm_final)

    # ---- 1. Per-dim std across samples (at each timestep) ----
    theta_std = np.nanstd(all_theta, axis=0)   # (K, D)
    theta_mean = np.nanmean(all_theta, axis=0)
    cv = theta_std / (np.abs(theta_mean) + 1e-30)  # coefficient of variation

    print(f"\n=== Theta variance across {len(test_idx)} test samples ===")
    print(f"{'dim':>4}  {'mean(mean)':>12}  {'mean(std)':>12}  {'mean(CV)':>10}  {'min':>10}  {'max':>10}")
    for d in range(all_theta.shape[-1]):
          vals = all_theta[:, :, d].flatten()
          print(f"{d:>4}  {np.nanmean(vals):>12.3e}  {np.nanstd(vals):>12.3e}  "
              f"{np.nanmean(cv[:, d]):>10.3f}  {np.nanmin(vals):>10.3e}  {np.nanmax(vals):>10.3e}")

    # ---- 2. Correlation: last-valid theta vs pm_final ----
    print(f"\n=== Corr(theta_final, pm_final) ===")
    lengths = np.array([th.shape[0] for th in all_theta])
    last_idx = np.clip(lengths - 1, 0, None)
    for d in range(all_theta.shape[-1]):
        th_last = all_theta[np.arange(len(all_theta)), last_idx, d]
        if np.all(np.isnan(th_last)):
            corr = np.nan
        else:
            corr = np.corrcoef(th_last, all_pm_final)[0, 1]
        print(f"  dim {d}: r = {corr:.3f}")

    # ---- 3. Plot: theta distributions per dim ----
    D = all_theta.shape[-1]
    fig, axes = plt.subplots(2, D, figsize=(D * 3, 6))
    for d in range(D):
        # top: box plot of theta_final per sample
        ax = axes[0, d]
        ax.boxplot(all_theta[:, :, d].T, showfliers=False)
        ax.set_title(f"theta[{d}] per sample")
        ax.set_xlabel("sample")
        ax.set_ylabel("theta")

        # bottom: scatter theta_final vs pm_final
        ax2 = axes[1, d]
        th_last = all_theta[np.arange(len(all_theta)), last_idx, d]
        ax2.scatter(th_last, all_pm_final, alpha=0.5, s=10)
        ax2.set_xlabel(f"theta[{d}] final")
        ax2.set_ylabel("pm final (true)")

    plt.tight_layout()
    out = exp_dir / "theta_variance.png"
    plt.savefig(out, dpi=120)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
