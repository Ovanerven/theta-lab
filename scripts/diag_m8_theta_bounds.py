"""
Offline diagnostic: forward-simulate M8 with fixed θ across [lo, hi] to find
what (if any) θ values produce realistic pm trajectories matching dataset
target magnitudes. If no θ in the bounds reaches pm ≈ 89, the theta bounds
themselves are wrong.

Also reports what the encoder's init (geomean of bounds) would produce, to
test the "geomean init too low for translation" hypothesis.

Run:
  /home/overven1/.conda/envs/thesis_env/bin/python scripts/diag_m8_theta_bounds.py
"""
import sys
import numpy as np

sys.path.insert(0, "last-layer-ode")
import torch
from scaffolds import SCAFFOLDS


def euler_rollout(rhs, y0, u_seq, jump, dt, theta, T_steps):
    """Plain Euler integration: y_{k+1} = y_k + dt * f(y_k, theta) + u_k @ jump."""
    y = y0.clone()
    traj = [y.clone()]
    for k in range(T_steps):
        # bolus deposit
        y = y + u_seq[k] @ jump
        # rhs step
        dy = rhs(y.unsqueeze(0), theta.unsqueeze(0))[0]
        y = y + dt * dy
        traj.append(y.clone())
    return torch.stack(traj, dim=0)  # (T+1, P)


def make_theta(scaffold, mode, lo_vec, hi_vec):
    """Build θ vector at various points in the bounds box."""
    lo = torch.tensor(lo_vec, dtype=torch.float32)
    hi = torch.tensor(hi_vec, dtype=torch.float32)
    if mode == "geomean":
        return torch.sqrt(lo * hi)
    if mode == "hi":
        return hi.clone()
    if mode == "lo":
        return lo.clone()
    if mode == "mid_lin":
        return 0.5 * (lo + hi)
    raise ValueError(mode)


def make_theta_smart(scaffold, lo_vec, hi_vec, name_to_idx, push_high_keys, push_low_keys):
    """θ at geomean except: push specified params to their hi (or lo) bound."""
    lo = torch.tensor(lo_vec, dtype=torch.float32)
    hi = torch.tensor(hi_vec, dtype=torch.float32)
    theta = torch.sqrt(lo * hi)
    for k in push_high_keys:
        if k in name_to_idx:
            theta[name_to_idx[k]] = hi[name_to_idx[k]]
    for k in push_low_keys:
        if k in name_to_idx:
            theta[name_to_idx[k]] = lo[name_to_idx[k]]
    return theta


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # ── Load dataset ──────────────────────────────────────────────────────
    z = np.load("datasets/cell-free/txtl_native_old_only.npz", allow_pickle=True)
    cn = [str(c) for c in z["control_names"]]
    y0_all = torch.tensor(z["y0"], dtype=torch.float32)           # (N, 7)
    u_all  = torch.tensor(z["u_seq"], dtype=torch.float32)         # (N, K, U)
    y_all  = torch.tensor(z["y_seq"], dtype=torch.float32)         # (N, K, 7)
    lengths = z["lengths"]
    dt_per_sample = z["dt_per_sample"]                              # (N,)
    obs_names = [str(s) for s in z["obs_names"]]
    print(f"dataset: N={y0_all.shape[0]}, K={y_all.shape[1]}, obs_names={obs_names}")
    print(f"control_names: {cn}")

    # pm is at obs idx 5
    pm_max_per_sample = y_all[:, :, 5].max(dim=1).values.numpy()
    pm_max_all = float(pm_max_per_sample.max())
    print(f"pm max in dataset: mean={pm_max_per_sample.mean():.1f}  max={pm_max_all:.1f}  std={pm_max_per_sample.std():.1f}")

    # ── Pick a sample with high pm (closer to the high-yield regime) ──────
    high_idx = int(np.argmax(pm_max_per_sample))
    print(f"\nUsing sample idx={high_idx} (pm_max={pm_max_per_sample[high_idx]:.1f})")

    # ── M8 scaffold ────────────────────────────────────────────────────────
    scaffold = SCAFFOLDS["txtl_model8_reagent_resource"]
    print(f"\nM8 state_names: {scaffold.state_names}")
    print(f"M8 P={scaffold.P}, theta_dim={scaffold.theta_dim}")
    print(f"M8 theta_lo: {scaffold.theta_lo_vec}")
    print(f"M8 theta_hi: {scaffold.theta_hi_vec}")

    theta_names = [
        "alpha_TX", "alpha_TL",
        "k_E", "k_A", "k_T", "k_W",
        "K_E", "K_A", "K_Mg", "K_K", "K_C", "K_W",
        "k_dm", "k_matm", "k_mt", "beta_W",
    ]
    assert len(theta_names) == scaffold.theta_dim
    name_to_idx = {n: i for i, n in enumerate(theta_names)}

    lo_vec = scaffold.theta_lo_vec
    hi_vec = scaffold.theta_hi_vec

    # ── Lift y0 from dataset (7) to scaffold (12) ─────────────────────────
    # M8 has no overlap with dataset names (no R, O, no m/mm/p/pm at matching
    # positions). Only DNA is shared via name → state 11. But DNA in dataset
    # starts at 0 anyway. So all 12 scaffold states start at 0.
    y0_full = torch.zeros(scaffold.P)

    # ── Build u_to_y_jump (U=13 → P=12) ────────────────────────────────────
    U = len(cn)
    jump = torch.zeros((U, scaffold.P), dtype=torch.float32)
    for j, name in enumerate(cn):
        target = scaffold.control_state_map.get(str(name).strip())
        if target is not None:
            jump[j, int(target)] = 1.0

    # ── Sample's actual length and dt ─────────────────────────────────────
    L = int(lengths[high_idx])
    dt_scalar = float(dt_per_sample[high_idx]) if dt_per_sample.ndim == 1 else 60.0
    u_seq = u_all[high_idx, :L, :]   # (L, U)
    print(f"\nSample length={L}, dt={dt_scalar}s, total time = {L*dt_scalar/60:.1f} min")

    # ── Forward-sim with various θ ─────────────────────────────────────────
    target_pm = float(y_all[high_idx, :L, 5].max())
    target_mm = float(y_all[high_idx, :L, 3].max())
    print(f"Target: mm_max={target_mm:.1f}, pm_max={target_pm:.1f}")

    rhs = scaffold.forward  # callable on (y, theta) batched

    configs = []
    configs.append(("geomean (encoder init)", make_theta(scaffold, "geomean", lo_vec, hi_vec)))
    configs.append(("all hi (max rates)",      make_theta(scaffold, "hi",      lo_vec, hi_vec)))
    configs.append(("all lo (min rates)",      make_theta(scaffold, "lo",      lo_vec, hi_vec)))

    # Smart picks: push rate constants high, decay low
    push_high = ["alpha_TX", "alpha_TL", "k_matm", "k_mt"]
    push_low  = ["k_dm", "k_E", "k_A", "k_T", "k_W"]
    configs.append((
        f"smart (high α/k_mt, low decays)",
        make_theta_smart(scaffold, lo_vec, hi_vec, name_to_idx, push_high, push_low),
    ))
    # Same but also push K's to their lo (gates → 1.0)
    push_low2 = push_low + ["K_E", "K_A", "K_Mg", "K_K", "K_C", "K_W"]
    configs.append((
        f"smart + K's at lo (gates saturated)",
        make_theta_smart(scaffold, lo_vec, hi_vec, name_to_idx, push_high, push_low2),
    ))

    print(f"\n{'config':<42}  {'mm_max':>10}  {'pm_max':>10}  {'verdict'}")
    print("─" * 95)
    for name, theta in configs:
        try:
            with torch.no_grad():
                traj = euler_rollout(rhs, y0_full, u_seq, jump, dt_scalar, theta, L)
            mm_pred_max = float(traj[:, 8].max())     # M8: mm @ state 8
            pm_pred_max = float(traj[:, 10].max())    # M8: pm @ state 10
            if not np.isfinite(pm_pred_max):
                verdict = "OVERFLOW (NaN/inf)"
            elif pm_pred_max < 0.5:
                verdict = "DEAD (pm≈0)"
            elif pm_pred_max < target_pm * 0.3:
                verdict = f"TOO LOW ({100*pm_pred_max/target_pm:.0f}% of target)"
            elif pm_pred_max > target_pm * 3:
                verdict = f"TOO HIGH ({100*pm_pred_max/target_pm:.0f}% of target)"
            else:
                verdict = f"IN RANGE ({100*pm_pred_max/target_pm:.0f}% of target) ✓"
            print(f"{name:<42}  {mm_pred_max:>10.2f}  {pm_pred_max:>10.2f}  {verdict}")
        except Exception as e:
            print(f"{name:<42}  ERROR: {e}")

    # ── Print geomean θ values numerically for inspection ─────────────────
    print("\nGeomean θ values (the encoder's sigmoid-bias-0 init):")
    theta_g = make_theta(scaffold, "geomean", lo_vec, hi_vec)
    for n, v, lo_v, hi_v in zip(theta_names, theta_g.tolist(), lo_vec, hi_vec):
        print(f"  {n:<10}: geomean={v:.3e}    [lo={lo_v:.0e}, hi={hi_v:.0e}]")

    # ── Quick reagent-state check at end of trajectory under geomean ──────
    with torch.no_grad():
        traj_g = euler_rollout(rhs, y0_full, u_seq, jump, dt_scalar, theta_g, L)
    print("\nReagent state magnitudes under geomean θ (final / max over trajectory):")
    for s_idx, s_name in enumerate(scaffold.state_names):
        print(f"  state[{s_idx}] {s_name:<6}: final={float(traj_g[-1, s_idx]):.3f}, max={float(traj_g[:, s_idx].max()):.3f}")


if __name__ == "__main__":
    main()
