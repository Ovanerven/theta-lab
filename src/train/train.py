from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.simple_ode_rnn import SimpleRNN
from src.data.ode_dataset import ODEDataset, collate

@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    hidden: int = 128
    num_layers: int = 1
    batch: int = 300
    decay: float = 0.0085
    val_frac: float = 0.15
    normalize: bool = False
    teacher_forcing: bool = True
    tf_drop_epoch: int = 250  # keep supervisor behavior (even if epochs<250)
    tf_every: int = 50
    grad_clip: float = 1.0
    save_path: Optional[str | Path] = None
    seed: int = 42
    log_dir: str = "logs"
    save_checkpoint_every: int = 0  # Save checkpoint every N epochs (0=disabled, saves space)

def Training_loop(
    dataset_path: str | Path,
    *,
    cfg: TrainConfig = TrainConfig(),
    model_cls=SimpleRNN,
):
    start_time = time.time()
    
    # --- reproducibility ---
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # --- device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- load dataset ---
    ds = ODEDataset(dataset_path)

    N = len(ds)
    idx = np.arange(N)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(idx)

    # --- split train/val (same logic as supervisor) ---
    if cfg.val_frac > 0.0 and N > 1:
        n_val = max(1, int(N * cfg.val_frac))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
    else:
        val_idx, train_idx = np.array([], dtype=int), idx

    train_ds = torch.utils.data.Subset(ds, train_idx.tolist())
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )

    val_loader = None
    if len(val_idx) > 0:
        val_ds = torch.utils.data.Subset(ds, val_idx.tolist())
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
            pin_memory=True,
        )

    # infer dimensions
    y0_ex, u_ex, dt_ex, y_ex = ds[0]
    P = int(y0_ex.shape[0])
    K_ex, U = (int(u_ex.shape[0]), int(u_ex.shape[1]))

    control = ds.control_indices.tolist()
    obs = ds.obs_indices.tolist()
    if len(control) != U:
        raise ValueError(f"Dataset control_indices has length {len(control)} but u_seq has U={U}")
    if len(obs) != P:
        raise ValueError(f"Dataset obs_indices has length {len(obs)} but y0 has P={P}")

    # Build (U,P) mapping: 1 if control species is observed species, else 0.
    jump = torch.zeros(U, P, dtype=torch.float32)
    obs_pos = {full_idx: p for p, full_idx in enumerate(obs)}
    for u, full_idx in enumerate(control):
        p = obs_pos.get(full_idx)
        if p is not None:
            jump[u, p] = 1.0

    model = model_cls(U, P=P, hidden=cfg.hidden, num_layers=cfg.num_layers, u_to_y_jump=jump).to(device)
    
    # optional compilation - disabled for MPS debugging
    do_script = False
    if do_script:
        try:
            model = torch.jit.script(model)
            print("The model compiled successfully")
        except Exception as e:
            print(f"TorchScript failed: {e}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.decay)

    best_val = float("inf")
    best_state = None

    # logging
    train_losses = []
    val_losses = []

    for ep in range(1, cfg.epochs + 1):
        teacher_forcing = cfg.teacher_forcing and (ep < cfg.tf_drop_epoch)
        if ep == cfg.tf_drop_epoch:
            teacher_forcing = False

        # ------------------------- TRAIN -------------------------
        model.train()
        train_total = 0.0
        n_batches = 0

        for y0, u_seq, dt_seq, y_seq in train_loader:
            y0 = y0.to(device)         # (B,P) initial observed state
            u_seq = u_seq.to(device)   # (B,K,U) control inputs
            dt_seq = dt_seq.to(device) # (B,K) time intervals
            y_seq = y_seq.to(device)   # (B,K,P) observed trajectory

            opt.zero_grad()

            pred, theta = model(
                y0, u_seq, dt_seq,
                y_seq=y_seq,
                teacher_forcing=teacher_forcing,
                tf_every=cfg.tf_every,
            )  # pred: (B,K,P)

            # NaN detection
            if not torch.isfinite(pred).all():
                print(f"WARNING: NaNs in pred at epoch {ep}, batch {n_batches+1}")
                print(f"  pred stats: min={pred.min()}, max={pred.max()}, has_nan={torch.isnan(pred).any()}, has_inf={torch.isinf(pred).any()}")
                raise RuntimeError("NaNs detected in predictions")

            # supervisor's approach: clamp at 1.0 before log1p to avoid near-zero regime
            y_clamped = y_seq.clamp_min(1.0)
            pred_clamped = pred.clamp_min(1.0)
            # y_clamped = y_seq
            # pred_clamped = pred
            log_y = torch.log1p(y_clamped)
            log_pred = torch.log1p(pred_clamped)

            loss = (log_pred - log_y).pow(2).mean()

            if not torch.isfinite(loss):
                print(f"WARNING: NaN/inf loss at epoch {ep}, batch {n_batches+1}")
                print(f"  loss value: {loss}")
                raise RuntimeError("NaN/inf detected in loss")

            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            train_total += float(loss.item())
            n_batches += 1

        train_loss = train_total / max(1, n_batches)
        train_losses.append(train_loss)

        # ------------------------- VAL -------------------------
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            v_batches = 0
            with torch.no_grad():
                for y0, u_seq, dt_seq, y_seq in val_loader:
                    y0 = y0.to(device)
                    u_seq = u_seq.to(device)
                    dt_seq = dt_seq.to(device)
                    y_seq = y_seq.to(device)

                    pred, _ = model(y0, u_seq, dt_seq, y_seq=None, teacher_forcing=False)
                    y_clamped = y_seq.clamp_min(1.0)
                    pred_clamped = pred.clamp_min(1.0)
                        
                    # y_clamped = y_seq
                    # pred_clamped = pred

                    loss = (torch.log1p(pred_clamped) - torch.log1p(y_clamped)).pow(2).mean()
                    val_total += float(loss.item())
                    v_batches += 1

            val_loss = val_total / max(1, v_batches)
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if val_loss is None:
            print(f"ep {ep:4d} | train {train_loss:.6f}")
        else:
            print(f"ep {ep:4d} | train {train_loss:.6f} | val {val_loss:.6f} | best {best_val:.6f}")
        
        # Save periodic checkpoints if requested
        if cfg.save_checkpoint_every > 0 and ep % cfg.save_checkpoint_every == 0:
            ckpt_dir = Path(cfg.log_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"model_ep{ep:04d}.pt"
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"state_dict": state, "epoch": ep, "cfg": cfg.__dict__}, ckpt_path)

    # save best model if requested
    if best_state is not None:
        model.load_state_dict(best_state)

    # save loss curves
    loss_dir = Path(cfg.log_dir)
    loss_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        loss_dir / "loss_curves.npz",
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses) if val_losses else None,
        config=cfg.__dict__,  # save full config for comparison
    )
    print(f"Saved loss curves to {loss_dir / 'loss_curves.npz'}")

    if cfg.save_path is not None:
        cfg_path = Path(cfg.save_path)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save({"state_dict": state, "cfg": cfg.__dict__, "best_val": best_val}, cfg_path)
        print(f"Saved best model to {cfg_path}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f}s ({elapsed/60:.2f}m)")

    return best_state


if __name__ == "__main__":
    # example:
    # python train_simple.py --data ode_dataset_train.npz
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ode_dataset_train.npz")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save", type=str, default=None, help="Model save path (auto-generated if not provided)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory name (auto-generated if not provided)")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Save checkpoint every N epochs (0=disabled)")
    args = parser.parse_args()

    # auto-generate timestamp for unique naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # auto-generate log directory name
    if args.log_dir is None:
        log_dir = f"logs/ep{args.epochs}_b{args.batch}_lr{args.lr}_seed{args.seed}_{timestamp}"
    else:
        log_dir = args.log_dir
    
    # auto-generate model save path
    if args.save is None:
        dataset_name = Path(args.data).stem
        save_path = f"models/{dataset_name}_ep{args.epochs}_b{args.batch}_lr{args.lr}_seed{args.seed}_{timestamp}.pt"
    else:
        save_path = args.save

    cfg = TrainConfig(
        epochs=args.epochs, 
        batch=args.batch, 
        lr=args.lr, 
        save_path=save_path, 
        seed=args.seed, 
        log_dir=log_dir,
        save_checkpoint_every=args.checkpoint_every,
    )
    Training_loop(args.data, cfg=cfg)