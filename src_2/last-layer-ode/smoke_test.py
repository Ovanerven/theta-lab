from __future__ import annotations

import os
import numpy as np
import torch

from create_dataset import generate_training_dataset
from scaffolds import Scaffold
from ode_rnn import ODERNN
from jumps import make_u_to_y_jump


def print_jump_structure(J: torch.Tensor, control_names, obs_names, control_indices, obs_indices) -> None:
    J = J.detach().cpu()
    control_indices = [int(x) for x in control_indices]
    obs_indices = [int(x) for x in obs_indices]

    print("\n=== u_to_y_jump structure ===")
    print(f"J shape: {tuple(J.shape)}  |  nnz: {(J != 0).sum().item()}")

    shown = 0
    for j in range(J.shape[0]):
        cols = (J[j] != 0).nonzero(as_tuple=False).view(-1).tolist()
        if not cols:
            continue
        for p in cols:
            c_full = control_indices[j]
            y_full = obs_indices[p]
            c_name = str(control_names[j]) if control_names is not None else f"u[{j}]"
            y_name = str(obs_names[p]) if obs_names is not None else f"y[{p}]"
            tag = "OK" if c_full == y_full else "MISMATCH"
            print(f"u[{j:02d}] {c_name:>6s} (full={c_full:02d}) -> y[{p:02d}] {y_name:>6s} (full={y_full:02d})  [{tag}]")
            shown += 1

    if shown == 0:
        print("No direct observed jumps (all-zero J). This is expected if controls target only unobserved species.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # 1) Generate tiny dataset
    # ----------------------------
    out_path = "tmp_smoke_dataset.npz"
    if os.path.exists(out_path):
        os.remove(out_path)

    generate_training_dataset(
        n_samples=3,
        t_span=10.0,
        n_steps=21,          # K = n_steps - 1 = 20
        control_indices=None, # default: all full species are controllable
        obs_indices=None,     # default: [0,3,6,9,12]
        zero_init=True,
        tail=0.0,
        output_file=out_path,
        seed=0,
        k_noise=0.0,
    )

    d = np.load(out_path, allow_pickle=True)

    y0_np = d["y0"]          # (N,P)
    u_np = d["u_seq"]        # (N,K,U)
    y_np = d["y_seq"]        # (N,K,P)
    t_obs = d["t_obs"]       # (K+1,)

    obs_indices = d["obs_indices"].tolist()
    control_indices = d["control_indices"].tolist()
    obs_names = d["obs_names"].tolist() if "obs_names" in d else None
    control_names = d["control_names"].tolist() if "control_names" in d else None

    N, K, P = y_np.shape
    U = u_np.shape[2]

    dt_np = np.diff(t_obs).astype(np.float32)          # (K,)
    dt_np = np.repeat(dt_np[None, :], N, axis=0)       # (N,K)

    print("=== dataset ===")
    print(f"y0:    {y0_np.shape}")
    print(f"u_seq: {u_np.shape}")
    print(f"y_seq: {y_np.shape}")
    print(f"dt:    {dt_np.shape}")
    print(f"obs_indices:     {obs_indices}")
    print(f"control_indices: {control_indices}")

    # ----------------------------
    # 2) Build jump matrix
    # ----------------------------
    J = make_u_to_y_jump(control_indices, obs_indices, device=device)
    print_jump_structure(J, control_names, obs_names, control_indices, obs_indices)

    # ----------------------------
    # 3) Instantiate model
    # ----------------------------
    # Use a minimal "identity" scaffold: dy/dt = 0 (so we only test plumbing).
    # This avoids needing to pick a specific mechanistic RHS for the smoke test.
    def rhs_zero(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(y)

    scaffold = Scaffold(
        P=P,
        theta_dim=8,  # arbitrary for smoke test
        state_names=tuple(obs_names) if obs_names is not None else tuple([f"y{i}" for i in range(P)]),
        rhs=rhs_zero,
    )

    model = ODERNN(
        U=U,
        scaffold=scaffold,
        u_to_y_jump=J,
        hidden=32,
        lift_dim=16,
        num_layers=1,
        dropout=0.0,
        theta_lo=1e-3,
        theta_hi=2.0,
        n_substeps=2,
    ).to(device)

    # ----------------------------
    # 4) Forward pass
    # ----------------------------
    y0 = torch.tensor(y0_np, device=device)
    u_seq = torch.tensor(u_np, device=device)
    dt_seq = torch.tensor(dt_np, device=device)

    with torch.no_grad():
        y_hat, theta = model(
            y0=y0,
            u_seq=u_seq,
            dt_seq=dt_seq,
            y_seq=torch.tensor(y_np, device=device),
            teacher_forcing=False,
        )

    print("\n=== forward pass ===")
    print(f"y_hat:  {tuple(y_hat.shape)}  (expected {(N, K, P)})")
    print(f"theta:  {tuple(theta.shape)} (expected {(N, K, scaffold.theta_dim)})")
    assert y_hat.shape == (N, K, P)
    assert theta.shape == (N, K, scaffold.theta_dim)

    # sanity: since rhs=0, only jumps change y. Check that at least something happened.
    delta = (y_hat[:, 0, :] - y0).abs().max().item()
    print(f"max |y_hat[:,0]-y0| = {delta:.6f}  (should be >0 if any u in first interval hits observed species)")
    print("\nSmoke test passed ✅")


if __name__ == "__main__":
    main()