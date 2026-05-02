from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from typing import Optional, Sequence

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.subplots as subplots
import matplotlib.pyplot as plt
import yaml

from train import ODEDataset, collate, collate_varlen, _apply_norm
from models import MODELS
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


def _test_subset(ds: ODEDataset, exp_dir: Path) -> ODEDataset | torch.utils.data.Subset:
    """Return the test split subset if split.npz exists, else the full dataset."""
    split_path = exp_dir / "split.npz"
    if split_path.exists():
        split = np.load(split_path)
        test_idx = split["test_idx"].tolist()
        if len(test_idx) > 0:
            return torch.utils.data.Subset(ds, test_idx)
    return ds


def rebuild_model_from_experiment(exp_dir: Path, device: torch.device, ckpt_path: Path | None = None) -> Tuple[torch.nn.Module, ODEDataset, list[str], list[str]]:
    cfg = load_yaml(exp_dir / "config.yaml")

    dataset_path = Path(cfg["dataset_path"])
    if not dataset_path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]  # .../theta-lab
        script_dir = Path(__file__).resolve().parent        # .../theta-lab/last-layer-ode
        candidates_paths = [
            dataset_path,                                         # relative to cwd
            exp_dir / dataset_path,                               # relative to exp_dir
            script_dir / dataset_path,                            # relative to this script
            project_root / dataset_path,                          # relative to project root
            Path(*dataset_path.parts[1:]) if dataset_path.parts[0] == ".." else None,  # strip leading ../
            Path("datasets") / dataset_path.name,                # just filename in datasets/
            script_dir / "datasets" / dataset_path.name,
            project_root / "datasets" / dataset_path.name,
        ]
        dataset_path = next(
            (p for p in candidates_paths if p is not None and p.exists()), None
        )
        if dataset_path is None:
            raise FileNotFoundError(
                f"Dataset not found: {cfg['dataset_path']} "
                f"(checked cwd, {exp_dir}, {script_dir}, and {project_root}/datasets)"
            )

    ds = ODEDataset(dataset_path)

    norm_path = exp_dir / "norm_stats.npz"
    if norm_path.exists():
        norm = np.load(norm_path, allow_pickle=True)
        method = str(norm["method"])
        mean = norm["mean"] if "mean" in norm else None
        std  = norm["std"]  if "std"  in norm else None
        ds.y0   = _apply_norm(ds.y0,   method, mean, std)
        ds.y_seq = _apply_norm(ds.y_seq, method, mean, std)

    scaffold_name = cfg.get("scaffold", "reduced5")
    scaffold = SCAFFOLDS[scaffold_name]

    # infer dims
    y0_ex, u_ex, _ = ds[0]
    P_obs = int(y0_ex.shape[0])
    U = int(u_ex.shape[-1])

    if scaffold.P != P_obs:
        raise ValueError(f"Scaffold {scaffold_name} expects P={scaffold.P}, but dataset has P_obs={P_obs}.")

    jump = make_u_to_y_jump(ds.control_indices, ds.obs_indices, device=device)

    # Recompute gru_u_cols / gru_y_cols from config (same logic as train_R.py)
    gru_u_cols = None
    if cfg.get("exclude_ode_cols_from_gru", False):
        gru_u_cols = [j for j in range(U) if int(ds.control_indices[j]) >= scaffold.P]

    gru_y_cols = None
    if cfg.get("gru_y_obs_only", False):
        obs_idx_cfg = cfg.get("obs_idx")
        gru_y_cols = list(obs_idx_cfg) if obs_idx_cfg is not None else list(range(scaffold.P))

    # Allow plotting any checkpoint (including in-progress runs).
    # Priority: explicit ckpt_path > model.pt (final best) > model_last.pt > newest checkpoints/ckpt_ep*.pt.
    if ckpt_path is not None:
        chosen = Path(ckpt_path)
    elif (exp_dir / "model.pt").exists():
        chosen = exp_dir / "model.pt"
    elif (exp_dir / "model_last.pt").exists():
        chosen = exp_dir / "model_last.pt"
    else:
        ckpts = sorted((exp_dir / "checkpoints").glob("ckpt_ep*.pt")) if (exp_dir / "checkpoints").exists() else []
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints in {exp_dir} (looked for model.pt, model_last.pt, checkpoints/ckpt_ep*.pt)")
        chosen = ckpts[-1]
    print(f"Loading checkpoint: {chosen}")
    ckpt = torch.load(chosen, map_location="cpu")

    model_class_name = cfg.get("model_class", "ode_rnn")
    ModelClass = MODELS[model_class_name]
    model = ModelClass(
        U=U,
        rhs=scaffold,
        u_to_y_jump=jump,
        hidden=int(cfg.get("hidden", 128)),
        lift_dim=int(cfg.get("lift_dim", 32)),
        num_layers=int(cfg.get("num_layers", 1)),
        dropout=float(cfg.get("dropout", 0.0)),
        theta_lo=float(cfg.get("theta_lo", 1e-3)),
        theta_hi=float(cfg.get("theta_hi", 2.0)),
        n_substeps=int(cfg.get("n_substeps", 1)),
        use_basal=bool(cfg.get("use_basal", False)),
        theta_bounded=bool(cfg.get("theta_bounded", True)),
        context_len=int(cfg.get("context_len", 64)),
        ff_mult=int(cfg.get("ff_mult", 2)),
        d_state=int(cfg.get("d_state", 16)),
        expand=int(cfg.get("expand", 2)),
        d_conv=int(cfg.get("d_conv", 4)),
        gru_u_cols=gru_u_cols,
        gru_y_cols=gru_y_cols,
        head_bias_init=float(cfg.get("head_bias_init", 0.0)),
        head_weight_gain=float(cfg.get("head_weight_gain", 1.0)),
        head_bottle_dims=cfg.get("head_bottle_dims", None),
        head_dims=cfg.get("head_dims", None),
        theta_head_transform=str(cfg.get("theta_head_transform", "log_gamma")),
        head_bottle=bool(cfg.get("head_bottle", False)),
        lift_skip=bool(cfg.get("lift_skip", False)),
    ).to(device)

    model.load_state_dict(ckpt["state_dict"], strict=False)
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


def plot_val_species_losses(loss_npz: Path, out_dir: Path, state_names: list[str], obs_idx: list[int] | None = None):
    data = np.load(loss_npz, allow_pickle=True)
    if "val_species_losses" not in data.files or data["val_species_losses"] is None:
        return
    V = data["val_species_losses"]
    if V.size == 0:
        return

    if obs_idx is not None and len(obs_idx) == V.shape[1]:
        obs_names = [state_names[i] for i in obs_idx]
    else:
        obs_names = state_names[:V.shape[1]]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(V.T, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Species")
    ax.set_title("Val per-species loss")
    ax.set_yticks(np.arange(len(obs_names)))
    ax.set_yticklabels(obs_names)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "val_species_heatmap.png", dpi=150)
    plt.close(fig)

    last = V[-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    if obs_idx is not None and len(obs_idx) == len(last):
        full = np.zeros(len(state_names))
        for i, idx in enumerate(obs_idx):
            full[idx] = last[i]
        ax.bar(np.arange(len(state_names)), full)
        ax.set_xticks(np.arange(len(state_names)))
        ax.set_xticklabels(state_names, rotation=0)
    else:
        ax.bar(np.arange(len(obs_names)), last)
        ax.set_xticks(np.arange(len(obs_names)))
        ax.set_xticklabels(obs_names, rotation=0)
    ax.set_xlabel("Species")
    ax.set_ylabel("Val loss")
    ax.set_title("Val per-species loss (final epoch)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "val_species_final.png", dpi=150)
    plt.close(fig)


def _save_bolus_panel(t, u_val, control_names, title, out_path):
    """Saves a standalone square-ish plot for the inputs/boluses."""
    U = u_val.shape[-1]
    fig, axes = plt.subplots(U, 1, figsize=(5, 4 * U), sharex=True)
    if U == 1:
        axes = [axes]
    
    for j, ax in enumerate(axes):
        ax.bar(t, u_val[:, j], width=(t[-1]-t[0])*0.015, color='C2', alpha=0.8)
        ax.set_ylabel(control_names[j])
        ax.grid(True, alpha=0.25)
    
    axes[-1].set_xlabel("Time")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_progression_dashboard(t, y_true, y_pred, u_val, theta_val, state_names, control_names, param_names, title, out_path):
    """Saves a strict 3-column Left-to-Right progression: Inputs -> Params -> States."""
    P = y_pred.shape[-1]
    U = u_val.shape[-1]
    D = theta_val.shape[-1] if theta_val is not None else 0

    n_rows = max(P, U, D)
    
    # 3 columns, perfectly square subplots: width=12 (3 cols * 4 in), height=4*n_rows
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_rows):
        # 1. Inputs (Left Column)
        ax_u = axes[i, 0]
        if i < U:
            ax_u.bar(t, u_val[:, i], width=(t[-1]-t[0])*0.02, color='C2', alpha=0.7)
            ax_u.set_title(f"Input: {control_names[i]}", fontsize=10, color='C2')
            ax_u.grid(True, alpha=0.25)
            ax_u.set_xlim([t[0], t[-1]])
            if i == U - 1:
                ax_u.set_xlabel("Time")
        else:
            ax_u.axis("off")

        # 2. Parameters / Theta (Middle Column)
        ax_th = axes[i, 1]
        if theta_val is not None and i < D:
            ax_th.plot(t, theta_val[:, i], linewidth=2, color='C3')
            name = param_names[i] if (param_names is not None and i < len(param_names)) else f"θ{i}"
            ax_th.set_title(f"Param: {name}", fontsize=10, color='C3')
            ax_th.grid(True, alpha=0.25)
            ax_th.set_xlim([t[0], t[-1]])
            if i == D - 1:
                ax_th.set_xlabel("Time")
        else:
            ax_th.axis("off")

        # 3. States / Trajectories (Right Column)
        ax_y = axes[i, 2]
        if i < P:
            ax_y.plot(t, y_true[:, i], linewidth=2, label="true", color='C0')
            ax_y.plot(t, y_pred[:, i], linewidth=2, linestyle="--", label="pred", color='C1')
            ax_y.set_title(f"State: {state_names[i]}", fontsize=10)
            ax_y.grid(True, alpha=0.25)
            ax_y.set_xlim([t[0], t[-1]])
            if i == 0:
                ax_y.legend(fontsize=8)
            if i == P - 1:
                ax_y.set_xlabel("Time")
        else:
            ax_y.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_predictions(model, ds: ODEDataset, state_names: list[str], out_dir: Path, n_samples: int, device: torch.device, exp_dir: Path | None = None, u_transform: str = "none", param_names: list[str] | None = None):
    raw_ds = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    plot_ds = _test_subset(ds, exp_dir) if exp_dir is not None else ds
    n_samples = min(int(n_samples), len(plot_ds))
    collate_fn = collate_varlen if getattr(raw_ds, "variable_length", False) else collate
    loader = torch.utils.data.DataLoader(plot_ds, batch_size=n_samples, shuffle=False, num_workers=0, collate_fn=collate_fn)
    split_label = "test" if (exp_dir is not None and (exp_dir / "split.npz").exists()) else "train"

    if isinstance(plot_ds, torch.utils.data.Subset):
        plotted_indices = list(plot_ds.indices[:n_samples])
    else:
        plotted_indices = list(range(n_samples))

    y0, u_seq, y_seq, batch_lengths = next(iter(loader))
    K_batch = u_seq.shape[1]
    dt_seq = torch.from_numpy(raw_ds.dt[:K_batch])[None, :].expand(y0.shape[0], -1)

    y0 = y0.to(device)
    u_seq = u_seq.to(device)
    y_seq = y_seq.to(device)
    dt_seq = dt_seq.to(device)
    if batch_lengths is not None:
        batch_lengths = batch_lengths.to(device)

    with torch.no_grad():
        obs_idx = torch.arange(y0.shape[-1], device=y0.device)
        model_kwargs = {"y_seq": None, "teacher_forcing": False, "u_transform": u_transform}
        if model.__class__.__name__ == "OdeTransformerGrouped" and batch_lengths is not None:
            model_kwargs["lengths"] = batch_lengths
        pred, _theta, _beta = model(y0, u_seq, dt_seq, obs_idx, **model_kwargs)

    y_true = y_seq.cpu().numpy()
    y_pred = pred.cpu().numpy()
    u_np = u_seq.cpu().numpy()
    dt_np = dt_seq.cpu().numpy()
    lengths_np = batch_lengths.cpu().numpy() if batch_lengths is not None else None
    theta_np = _theta.cpu().numpy() if _theta is not None else None

    if getattr(raw_ds, "control_names", None) is not None:
        control_names = raw_ds.control_names.tolist()
    else:
        control_names = [f"u{j}" for j in range(u_np.shape[-1])]

    P = y_pred.shape[-1]
    for i in range(n_samples):
        Li = int(lengths_np[i]) if lengths_np is not None else y_pred.shape[1]
        t = np.concatenate([[0.0], np.cumsum(dt_np[i, :Li])])[1:] 
        sample_id_str = f"{i:03d}_idx{plotted_indices[i]}"
        
        y_true_i = y_true[i, :Li, :]
        y_pred_i = y_pred[i, :Li, :]
        u_val_i = u_np[i, :Li, :]
        theta_val_i = theta_np[i, :Li, :] if theta_np is not None else None

        # ---------------------------------------------------------
        # Plot 1: Trajectories (now scaled to be perfectly square)
        # ---------------------------------------------------------
        fig, axes = plt.subplots(P, 1, figsize=(6, 4 * P), sharex=True)
        if P == 1:
            axes = [axes]
        for p, ax in enumerate(axes):
            ax.plot(t, y_true_i[:, p], linewidth=2, label="true")
            ax.plot(t, y_pred_i[:, p], linewidth=2, linestyle="--", label="pred")
            ax.set_ylabel(state_names[p] if p < len(state_names) else f"s{p}")
            ax.grid(True, alpha=0.25)
            if p == 0:
                ax.legend()
        axes[-1].set_xlabel("Time")
        fig.suptitle(f"Prediction vs truth [{split_label}] (sample {sample_id_str})")
        fig.tight_layout()
        fig.savefig(out_dir / f"pred_vs_true_{sample_id_str}.png", dpi=150)
        plt.close(fig)

        # ---------------------------------------------------------
        # Plot 2: Boluses / Inputs
        # ---------------------------------------------------------
        _save_bolus_panel(
            t, u_val_i, control_names, 
            title=f"Inputs/Boluses [{split_label}] (sample {sample_id_str})", 
            out_path=out_dir / f"inputs_bolus_{sample_id_str}.png"
        )

        # ---------------------------------------------------------
        # Plot 3: Parameters (Theta)
        # ---------------------------------------------------------
        if theta_np is not None:
            _save_theta_panel(
                theta_val_i, dt_np[i, :Li],
                out_dir / f"theta_sample_{sample_id_str}.png",
                title=f"Learned θ(t) [{split_label}] (sample {sample_id_str})",
                param_names=param_names,
            )

        # ---------------------------------------------------------
        # Plot 4: Combined Progression Dashboard
        # ---------------------------------------------------------
        _save_progression_dashboard(
            t, y_true_i, y_pred_i, u_val_i, theta_val_i, 
            state_names, control_names, param_names, 
            title=f"Progression Overview [{split_label}] (sample {sample_id_str})", 
            out_path=out_dir / f"progression_all_{sample_id_str}.png"
        )

    if theta_np is None:
        print("[warn] model returned no theta; skipping theta plots for plotted samples")


def _save_theta_panel(
    theta_np: np.ndarray,
    dt: np.ndarray,
    out_path: Path,
    title: str,
    param_names: list[str] | None = None,
):
    """Render a 2-column grid of θ_j(t), forced to be square."""
    t = np.concatenate([[0.0], np.cumsum(dt)])
    tt = t[1:]
    _, D = theta_np.shape
    n_cols = 2
    n_rows = (D + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 4 * n_rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for j, ax in enumerate(axes):
        if j < D:
            ax.plot(tt, theta_np[:, j], linewidth=1.8)
            name = param_names[j] if (param_names is not None and j < len(param_names)) else f"θ{j}"
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.25)
        else:
            ax.axis("off")

    axes[min(D - 1, len(axes) - 1)].set_xlabel("Time")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_beta(model, ds: ODEDataset, state_names: list[str], out_dir: Path, sample_idx: int, device: torch.device, exp_dir: Path | None = None, u_transform: str = "none"):
    """Plot the learned beta(t) residual terms."""
    sample_idx = int(sample_idx)
    raw_ds = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    plot_ds = _test_subset(ds, exp_dir) if exp_dir is not None else ds
    y0, u_seq, _ = plot_ds[sample_idx]
    K = int(u_seq.shape[0])
    dt_seq = torch.from_numpy(raw_ds.dt[:K])

    y0 = y0.unsqueeze(0).to(device)
    u_seq = u_seq.unsqueeze(0).to(device)
    dt_seq = dt_seq.unsqueeze(0).to(device)

    with torch.no_grad():
        obs_idx = torch.arange(y0.shape[-1], device=y0.device)
        model_kwargs = {"y_seq": None, "teacher_forcing": False, "u_transform": u_transform}
        if model.__class__.__name__ == "OdeTransformerGrouped":
            model_kwargs["lengths"] = torch.tensor([K], device=y0.device, dtype=torch.long)
        _, _theta, beta = model(y0, u_seq, dt_seq, obs_idx, **model_kwargs)

    beta_np = beta[0].cpu().numpy()  # (K, P)
    dt = dt_seq[0].cpu().numpy()
    t = np.concatenate([[0.0], np.cumsum(dt)])
    tt = t[1:]

    P = beta_np.shape[1]
    fig, axes = plt.subplots(P, 1, figsize=(6, 4 * P), sharex=True)
    if P == 1:
        axes = [axes]

    for p, ax in enumerate(axes):
        name = state_names[p] if p < len(state_names) else f"y{p}"
        ax.plot(tt, beta_np[:, p], linewidth=1.8)
        ax.axhline(0.0, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_ylabel(f"β({name})")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Learned β(t) residuals (sample {sample_idx})")
    fig.tight_layout()
    fig.savefig(out_dir / f"beta_sample{sample_idx}.png", dpi=150)
    plt.close(fig)


def plot_experiment(exp_dir: str | Path, n_samples: int = 5, sample_idx: int = 0, ckpt_path: str | Path | None = None) -> Path:
    exp_dir = Path(exp_dir)
    out_dir = exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device_auto()
    model, ds, state_names, param_names = rebuild_model_from_experiment(exp_dir, device=device, ckpt_path=ckpt_path)

    cfg = load_yaml(exp_dir / "config.yaml")
    obs_idx = list(cfg.get("obs_idx", [])) or None
    u_transform = str(cfg.get("u_transform", "none"))

    loss_npz = exp_dir / "logs" / "loss_curves.npz"
    if loss_npz.exists():
        plot_loss_curves(loss_npz, out_dir)
        plot_val_species_losses(loss_npz, out_dir, state_names=state_names, obs_idx=obs_idx)

    plot_predictions(model, ds, state_names=state_names, out_dir=out_dir, n_samples=n_samples, device=device, exp_dir=exp_dir, u_transform=u_transform, param_names=param_names)

    if cfg.get("use_basal", False):
        plot_beta(model, ds, state_names=state_names, out_dir=out_dir, sample_idx=sample_idx, device=device, exp_dir=exp_dir, u_transform=u_transform)

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

    model, ds, state_names, _param_names = rebuild_model_from_experiment(exp_dir, device=device)
    cfg_overlay = load_yaml(exp_dir / "config.yaml")
    u_transform_overlay = str(cfg_overlay.get("u_transform", "none"))

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
    raw_ds = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    y0, u_seq, y_seq = ds[sample_idx]
    K = int(u_seq.shape[0])
    dt = raw_ds.dt[:K].astype(np.float32)
    t = np.cumsum(dt)

    y0_b = y0.unsqueeze(0).to(device)
    u_b = u_seq.unsqueeze(0).to(device)
    dt_b = torch.from_numpy(raw_ds.dt[:K]).unsqueeze(0).to(device)

    y_true = y_seq.cpu().numpy()
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
            obs_idx = torch.arange(y0_b.shape[-1], device=y0_b.device)
            model_kwargs = {"y_seq": None, "teacher_forcing": False, "u_transform": u_transform_overlay}
            if model.__class__.__name__ == "OdeTransformerGrouped":
                model_kwargs["lengths"] = torch.tensor([K], device=y0_b.device, dtype=torch.long)
            pred, _theta, _beta = model(y0_b, u_b, dt_b, obs_idx, **model_kwargs)
        preds[ep] = pred[0].detach().cpu().numpy()

    fig, axes = plt.subplots(P, 1, figsize=(6, 4 * P), sharex=True)
    if P == 1:
        axes = [axes]

    n_chosen = len(chosen)
    for p in range(P):
        ax = axes[p]
        name = state_names[p] if p < len(state_names) else f"y{p}"

        ax.plot(t, y_true[:, p], linewidth=2.2, label="true")
        for i, ep in enumerate(sorted(chosen)):
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

    parser.add_argument("--epoch-overlays", action="store_true")
    parser.add_argument("--epochs", type=str, default=None, help='Comma list like "10,20,50"')
    parser.add_argument("--max-overlays", type=int, default=8)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Specific checkpoint path (default: model.pt > model_last.pt > newest checkpoints/ckpt_ep*.pt)")
    args = parser.parse_args()

    # 1) normal plots (loss curves, pred vs true, theta)
    plot_experiment(args.exp_dir, n_samples=args.n_samples, sample_idx=args.sample_idx, ckpt_path=args.ckpt)

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