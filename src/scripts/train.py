from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.data.ode_dataset import ODEDataset, collate
from src.sim.mechanisms import MECH
from src.models.kinetics_rnn import KineticsRNN


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (torch.log1p(pred) - torch.log1p(target)).pow(2).mean()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_from_config(config_path: str | Path) -> None:
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    t_start = time.time()

    ds_cfg   = cfg["dataset"]
    mech_cfg = cfg["mechanism"]
    tr_cfg   = cfg.get("train", {})
    out_cfg  = cfg.get("output", {})

    seed = int(tr_cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = pick_device()
    print(f"Using device: {device}")

    mech_name = mech_cfg["name"]
    mech = MECH[mech_name]
    print(f"Mechanism: {mech_name} | n_state={mech.n_state} n_param={mech.n_param}")

    dataset_path = Path(ds_cfg["path"])
    ds = ODEDataset(dataset_path, obs_indices=ds_cfg.get("obs_indices", None))
    print(f"Dataset: {dataset_path.name} | N={len(ds)} P={ds.y0.shape[1]} K={ds.u_seq.shape[1]}")
    print(f"obs_indices (effective): {ds.obs_indices.tolist()}")

    N = len(ds)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    val_frac = float(tr_cfg.get("val_frac", 0.15))
    n_val = max(1, int(N * val_frac)) if (val_frac > 0 and N > 1) else 0
    val_idx   = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()

    batch = int(tr_cfg.get("batch", 300))
    train_loader = DataLoader(
        torch.utils.data.Subset(ds, train_idx), batch_size=batch,
        shuffle=True, num_workers=0, collate_fn=collate, pin_memory=True,
    )
    val_loader = None
    if n_val > 0:
        val_loader = DataLoader(
            torch.utils.data.Subset(ds, val_idx), batch_size=batch,
            shuffle=False, num_workers=0, collate_fn=collate, pin_memory=True,
        )

    y0_ex, u_ex, _, _ = ds[0]
    P = int(y0_ex.shape[0])
    U = int(u_ex.shape[1])
    obs_full  = ds.obs_indices.tolist()
    ctrl_full = ds.control_indices.tolist()

    obs_pos = {full_idx: p for p, full_idx in enumerate(obs_full)}
    jump = torch.zeros(U, P, dtype=torch.float32)
    for u_i, full_idx in enumerate(ctrl_full):
        p = obs_pos.get(full_idx)
        if p is not None:
            jump[u_i, p] = 1.0

    model = KineticsRNN(
        mech=mech,
        obs_indices=obs_full,
        in_u=U,
        u_to_y_jump=jump,
        hidden=int(tr_cfg.get("hidden", 128)),
        num_layers=int(tr_cfg.get("num_layers", 1)),
        dropout=float(tr_cfg.get("dropout", 0.0)),
        n_sub=int(tr_cfg.get("n_sub", 1)),
    ).to(device)

    if tr_cfg.get("jit_script", True):
        try:
            model = torch.jit.script(model)
            print("TorchScript: OK")
        except Exception as e:
            print(f"TorchScript failed (running eagerly): {e}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr_cfg.get("lr", 1e-3)),
        weight_decay=float(tr_cfg.get("weight_decay", 0.0085)),
    )

    epochs       = int(tr_cfg.get("epochs", 200))
    grad_clip    = float(tr_cfg.get("grad_clip", 1.0))
    tf_every     = int(tr_cfg.get("tf_every", 50))
    tf_drop_ep   = int(tr_cfg.get("tf_drop_epoch", 250))
    tf_enabled   = bool(tr_cfg.get("teacher_forcing", True))
    ckpt_every   = int(out_cfg.get("checkpoint_every", 0))

    exp_name = out_cfg.get("exp_name", config_path.stem)
    exp_dir  = Path("experiments") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_dir  = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    model_path = exp_dir / "model.pt"

    best_val   = float("inf")
    best_state = None
    train_losses: list[float] = []
    val_losses:   list[float] = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        use_tf = tf_enabled and (ep < tf_drop_ep)

        model.train()
        total, nb = 0.0, 0
        for y0, u_seq, dt_seq, y_seq in train_loader:
            y0     = y0.to(device)
            u_seq  = u_seq.to(device)
            dt_seq = dt_seq.to(device)
            y_seq  = y_seq.to(device)

            opt.zero_grad()
            pred, _ = model(y0, u_seq, dt_seq, y_seq=y_seq, teacher_forcing=use_tf, tf_every=tf_every)
            loss = loss_fn(pred, y_seq)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total += float(loss.item())
            nb += 1

        train_losses.append(total / max(1, nb))

        va_loss = None
        if val_loader is not None:
            model.eval()
            vtotal, vb = 0.0, 0
            with torch.no_grad():
                for y0, u_seq, dt_seq, y_seq in val_loader:
                    y0     = y0.to(device)
                    u_seq  = u_seq.to(device)
                    dt_seq = dt_seq.to(device)
                    y_seq  = y_seq.to(device)
                    pred, _ = model(y0, u_seq, dt_seq, teacher_forcing=False)
                    vtotal += float(loss_fn(pred, y_seq).item())
                    vb += 1
            va_loss = vtotal / max(1, vb)
            val_losses.append(va_loss)
            if va_loss < best_val:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        dt = time.time() - t0
        if va_loss is None:
            print(f"ep {ep:4d} | train {train_losses[-1]:.6f} | {dt:.2f}s")
        else:
            print(f"ep {ep:4d} | train {train_losses[-1]:.6f} | val {va_loss:.6f} | best {best_val:.6f} | {dt:.2f}s")

        if ckpt_every > 0 and ep % ckpt_every == 0:
            ckpt_dir = log_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            state = {k.replace("_orig_mod.", ""): v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"state_dict": state, "epoch": ep, "mechanism": mech_name,
                        "obs_indices": obs_full, "cfg": cfg}, ckpt_dir / f"model_ep{ep:04d}.pt")

    if best_state is not None:
        model.load_state_dict(best_state)

    np.savez(log_dir / "loss_curves.npz",
             train_losses=np.array(train_losses),
             val_losses=np.array(val_losses) if val_losses else np.array([]))

    final_state = {k.replace("_orig_mod.", ""): v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": final_state, "mechanism": mech_name,
                "obs_indices": obs_full, "cfg": cfg, "best_val": best_val}, model_path)

    (exp_dir / "config_used.yaml").write_text(yaml.dump(cfg, sort_keys=False))
    print(f"Saved model → {model_path}")
    print(f"Done in {time.time() - t_start:.1f}s")

    try:
        from src.scripts.plotting import plot_loss_curves, plot_predictions, plot_learned_parameters, plot_evolution
        plots_dir = exp_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_loss_curves(log_dir / "loss_curves.npz", plots_dir)
        plot_predictions(model_path, dataset_path, plots_dir)
        plot_learned_parameters(str(model_path), str(dataset_path), output_path=str(plots_dir / "theta_sample0.png"))
        ckpt_dir = log_dir / "checkpoints"
        if ckpt_dir.exists() and any(ckpt_dir.glob("model_ep*.pt")):
            plot_evolution(str(log_dir), str(dataset_path), out_dir=str(plots_dir))
        print(f"Plots saved → {plots_dir}")
    except Exception as e:
        print(f"Auto-plotting failed (skipping): {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train_from_config(args.config)
