from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from typing import Optional, Sequence

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from train import ODEDataset, collate
from ode_rnn import ODERNN
from scaffolds import SCAFFOLDS
from jumps import make_u_to_y_jump


@dataclass
class PlotConfig:
    n_samples: int = 5
    sample_idx: int = 0


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def rebuild_model_from_experiment(exp_dir: Path, device: torch.device) -> Tuple[torch.nn.Module, ODEDataset, list[str], list[str]]:
    cfg = load_yaml(exp_dir / "config.yaml")

    ds = ODEDataset(cfg["dataset_path"])

    scaffold_name = cfg.get("scaffold", "reduced5")
    scaffold = SCAFFOLDS[scaffold_name]

    # infer dims
    y0_ex, u_ex, _ = ds[0]
    P_obs = int(y0_ex.shape[0])
    U = int(u_ex.shape[-1])

    if scaffold.P != P_obs:
        raise ValueError(f"Scaffold {scaffold_name} expects P={scaffold.P}, but dataset has P_obs={P_obs}.")

    jump = make_u_to_y_jump(ds.control_indices, ds.obs_indices, device=device)

    model = ODERNN(
        U=U,
        scaffold=scaffold,
        hidden=int(cfg.get("hidden", 128)),
        lift_dim=int(cfg.get("lift_dim", 32)),
        num_layers=int(cfg.get("num_layers", 1)),
        dropout=float(cfg.get("dropout", 0.0)),
        u_to_y_jump=jump,
        theta_lo=float(cfg.get("theta_lo", 1e-3)),
        theta_hi=float(cfg.get("theta_hi", 2.0)),
        n_substeps=int(cfg.get("n_substeps", 1)),
    ).to(device)

    ckpt = torch.load(exp_dir / "model.pt", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    state_names = ds.obs_names.tolist() if ds.obs_names is not None else [f"y{i}" for i in range(P_obs)]
    param_names = scaffold.param_names() if hasattr(scaffold, "param_names") else [f"θ{i}" for i in range(scaffold.theta_dim)]

    return model, ds, state_names, param_names


def plot_loss_curves(loss_npz: Path, out_dir: Path):
    data = np.load(loss_npz, allow_pickle=True)
    train_losses = data["train_losses"]
    val_losses = data["val_losses"] if "val_losses" in data.files and data["val_losses"] is not None else None

    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, label="train", linewidth=2)
    if val_losses is not None and len(val_losses) > 0:
        ax.plot(epochs, val_losses, label="val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log1p space)")
    ax.set_title("Loss curves")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close(fig)


def plot_val_species_losses(loss_npz: Path, out_dir: Path, state_names: list[str]):
    data = np.load(loss_npz, allow_pickle=True)
    if "val_species_losses" not in data.files or data["val_species_losses"] is None:
        return
    V = data["val_species_losses"]
    if V.size == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(V.T, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Species")
    ax.set_title("Val per-species loss (log1p-MSE)")
    ax.set_yticks(np.arange(len(state_names)))
    ax.set_yticklabels(state_names)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "val_species_heatmap.png", dpi=150)
    plt.close(fig)

    last = V[-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(last)), last)
    ax.set_xticks(np.arange(len(state_names)))
    ax.set_xticklabels(state_names, rotation=0)
    ax.set_xlabel("Species")
    ax.set_ylabel("Val loss")
    ax.set_title("Val per-species loss (final epoch)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "val_species_final.png", dpi=150)
    plt.close(fig)


def plot_predictions(model, ds: ODEDataset, state_names: list[str], out_dir: Path, n_samples: int, device: torch.device):
    n_samples = min(int(n_samples), len(ds))
    loader = torch.utils.data.DataLoader(ds, batch_size=n_samples, shuffle=False, num_workers=0, collate_fn=collate)

    y0, u_seq, y_seq = next(iter(loader))
    dt_seq = torch.from_numpy(ds.dt)[None, :].expand(y0.shape[0], -1)

    y0 = y0.to(device)
    u_seq = u_seq.to(device)
    y_seq = y_seq.to(device)
    dt_seq = dt_seq.to(device)

    with torch.no_grad():
        pred, _ = model(y0, u_seq, dt_seq, y_seq=None, teacher_forcing=False)

    y_true = y_seq.cpu().numpy()
    y_pred = pred.cpu().numpy()
    dt_np = dt_seq.cpu().numpy()

    P = y_pred.shape[-1]
    for i in range(n_samples):
        t = np.concatenate([[0.0], np.cumsum(dt_np[i])])
        fig, axes = plt.subplots(P, 1, figsize=(11, max(6, 2.0 * P)), sharex=True)
        if P == 1:
            axes = [axes]

        for p, ax in enumerate(axes):
            ax.plot(t[1:], y_true[i, :, p], linewidth=2, label="true")
            ax.plot(t[1:], y_pred[i, :, p], linewidth=2, linestyle="--", label="pred")
            ax.set_ylabel(state_names[p] if p < len(state_names) else f"s{p}")
            ax.grid(True, alpha=0.25)
            if p == 0:
                ax.legend()

        axes[-1].set_xlabel("Time")
        fig.suptitle(f"Prediction vs truth (sample {i})")
        fig.tight_layout()
        fig.savefig(out_dir / f"pred_vs_true_{i:03d}.png", dpi=150)
        plt.close(fig)


def plot_theta(model, ds: ODEDataset, param_names: list[str], out_dir: Path, sample_idx: int, device: torch.device):
    sample_idx = int(sample_idx)
    y0, u_seq, _ = ds[sample_idx]
    dt_seq = torch.from_numpy(ds.dt)

    y0 = y0.unsqueeze(0).to(device)
    u_seq = u_seq.unsqueeze(0).to(device)
    dt_seq = dt_seq.unsqueeze(0).to(device)

    with torch.no_grad():
        _, theta = model(y0, u_seq, dt_seq, y_seq=None, teacher_forcing=False)

    theta_np = theta[0].cpu().numpy()  # (K, theta_dim)
    dt = dt_seq[0].cpu().numpy()
    t = np.concatenate([[0.0], np.cumsum(dt)])
    tt = t[1:]

    _, D = theta_np.shape
    n_cols = 2
    n_rows = (D + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.2 * n_rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for j, ax in enumerate(axes):
        if j < D:
            ax.plot(tt, theta_np[:, j], linewidth=1.8)
            name = param_names[j] if j < len(param_names) else f"θ{j}"
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.25)
        else:
            ax.axis("off")

    axes[min(D - 1, len(axes) - 1)].set_xlabel("Time")
    fig.suptitle(f"Learned θ(t) (sample {sample_idx})")
    fig.tight_layout()
    fig.savefig(out_dir / f"theta_sample{sample_idx}.png", dpi=150)
    plt.close(fig)


def plot_experiment(exp_dir: str | Path, n_samples: int = 5, sample_idx: int = 0) -> Path:
    exp_dir = Path(exp_dir)
    out_dir = exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device_auto()
    model, ds, state_names, param_names = rebuild_model_from_experiment(exp_dir, device=device)

    loss_npz = exp_dir / "logs" / "loss_curves.npz"
    if loss_npz.exists():
        plot_loss_curves(loss_npz, out_dir)
        plot_val_species_losses(loss_npz, out_dir, state_names=state_names)

    plot_predictions(model, ds, state_names=state_names, out_dir=out_dir, n_samples=n_samples, device=device)
    plot_theta(model, ds, param_names=param_names, out_dir=out_dir, sample_idx=sample_idx, device=device)

    print(f"Saved plots to {out_dir}")
    return out_dir

def plot_epoch_prediction_overlays(
    exp_dir: str | Path,
    *,
    sample_idx: int = 0,
    epochs: list[int] | None = None,
    max_overlays: int = 8,
    out_name: str | None = None,
) -> Path:
    import re

    exp_dir = Path(exp_dir)
    out_dir = exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device_auto()

    # This loads config + dataset + builds model; it also loads model.pt (fine, we overwrite next)
    model, ds, state_names, _param_names = rebuild_model_from_experiment(exp_dir, device=device)

    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints/ folder found at: {ckpt_dir}")

    ckpt_files = sorted(ckpt_dir.glob("ckpt_ep*.pt"))
    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"No checkpoint files matching ckpt_ep*.pt in: {ckpt_dir}")

    epoch_to_path: dict[int, Path] = {}
    pat = re.compile(r"ckpt_ep(\d+)\.pt$")
    for p in ckpt_files:
        m = pat.search(p.name)
        if m:
            epoch_to_path[int(m.group(1))] = p

    if not epoch_to_path:
        raise RuntimeError(f"Could not parse any epochs from files in {ckpt_dir} (expected ckpt_epXXXX.pt).")

    available_epochs = sorted(epoch_to_path.keys())

    if epochs is None:
        if len(available_epochs) <= max_overlays:
            chosen = available_epochs
        else:
            idx = np.linspace(0, len(available_epochs) - 1, num=max_overlays)
            chosen = [available_epochs[int(round(i))] for i in idx]
            chosen = sorted(set(chosen))
    else:
        missing = [e for e in epochs if e not in epoch_to_path]
        if missing:
            raise ValueError(f"Requested epochs not found: {missing}\nAvailable: {available_epochs}")
        chosen = list(epochs)

    sample_idx = int(sample_idx)
    y0, u_seq, y_seq = ds[sample_idx]  # y_seq is (K,P) at t1..tK
    dt = ds.dt.astype(np.float32)      # (K,)
    t = np.cumsum(dt)                  # (K,) -> times for y_seq points (t1..tK)

    y0_b = y0.unsqueeze(0).to(device)
    u_b = u_seq.unsqueeze(0).to(device)
    dt_b = torch.from_numpy(ds.dt).unsqueeze(0).to(device)  # (1,K)

    y_true = y_seq.cpu().numpy()  # (K,P)
    P = int(y_true.shape[1])

    preds: dict[int, np.ndarray] = {}
    for ep in chosen:
        ckpt_path = epoch_to_path[ep]
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" not in ckpt:
            raise RuntimeError(f"Checkpoint missing 'state_dict': {ckpt_path}")

        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        with torch.no_grad():
            pred, _ = model(y0_b, u_b, dt_b, y_seq=None, teacher_forcing=False)
        preds[ep] = pred[0].detach().cpu().numpy()  # (K,P)

    fig_h = max(6.0, 2.0 * P)
    fig, axes = plt.subplots(P, 1, figsize=(11, fig_h), sharex=True)
    if P == 1:
        axes = [axes]

    n_chosen = len(chosen)
    for p in range(P):
        ax = axes[p]
        name = state_names[p] if p < len(state_names) else f"y{p}"

        ax.plot(t, y_true[:, p], linewidth=2.2, label="true")
        # Sort epochs so that alpha increases with epoch (newer = less transparent)
        for i, ep in enumerate(sorted(chosen)):
            # Alpha: earlier = more transparent, newer = less
            alpha = 0.3 + 0.7 * (i / max(n_chosen - 1, 1))
            ax.plot(t, preds[ep][:, p], linewidth=1.6, alpha=alpha, label=f"ep{ep:04d}")

        ax.set_ylabel(str(name))
        ax.grid(True, alpha=0.25)
        if p == 0:
            ax.legend(ncol=2, fontsize=9)

    axes[-1].set_xlabel("Time")
    title_epochs = ", ".join(str(e) for e in chosen)
    fig.suptitle(f"Predictions across epochs (sample {sample_idx}) | epochs: {title_epochs}")
    fig.tight_layout()

    if out_name is None:
        out_name = f"pred_overlays_sample{sample_idx:03d}.png"
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[epoch overlay] saved: {out_path}")
    return out_path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=str, help="experiments/<timestamp>_<name>")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--sample-idx", type=int, default=0)

    # optional: epoch overlay plot
    parser.add_argument("--epoch-overlays", action="store_true")
    parser.add_argument("--epochs", type=str, default=None, help='Comma list like "10,20,50"')
    parser.add_argument("--max-overlays", type=int, default=8)
    args = parser.parse_args()

    # 1) normal plots (loss curves, pred vs true, theta)
    plot_experiment(args.exp_dir, n_samples=args.n_samples, sample_idx=args.sample_idx)

    # 2) optional overlays from checkpoints
    if args.epoch_overlays:
        epochs = None
        if args.epochs is not None:
            epochs = [int(x.strip()) for x in args.epochs.split(",") if x.strip()]
        plot_epoch_prediction_overlays(
            args.exp_dir,
            sample_idx=args.sample_idx,
            epochs=epochs,
            max_overlays=args.max_overlays,
        )