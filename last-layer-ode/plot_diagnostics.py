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
import inspect

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


def _filter_model_kwargs(model: torch.nn.Module, kwargs: dict) -> dict:
    """Return a copy of kwargs containing only keys accepted by model.forward.

    This prevents passing unknown keyword arguments (like `u_transform`) to
    model.forward implementations that don't accept them.
    """
    try:
        sig = inspect.signature(model.forward)
        valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return valid
    except (ValueError, TypeError):
        return kwargs


# Mirror of train._LIFT_NAME_ALIASES. Kept in sync by hand — small enough.
_LIFT_NAME_ALIASES: dict[str, str] = {
    "O2": "O",
}


def _maybe_lift(y0: torch.Tensor, y_seq: torch.Tensor, lift_info: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror of train._lift_to_scaffold_state. No-op if lift_info is empty.

    Also performs the by-name IC copy for non-observed scaffold states whose
    names match dataset columns (with _LIFT_NAME_ALIASES). Without this,
    eval/plot would zero-init R/O for M5/M7/M9 and the rebuilt model would
    produce silent-zero predictions even though the trained checkpoint is fine.
    """
    if not lift_info:
        return y0, y_seq
    B, K = y0.shape[0], y_seq.shape[1]
    src = torch.as_tensor(lift_info["dataset_obs_idx"], device=y0.device, dtype=torch.long)
    dst = torch.as_tensor(lift_info["scaffold_obs_idx"], device=y0.device, dtype=torch.long)
    sP = int(lift_info["scaffold_P"])
    y0_full = torch.zeros((B, sP), device=y0.device, dtype=y0.dtype)
    y_seq_full = torch.zeros((B, K, sP), device=y_seq.device, dtype=y_seq.dtype)
    y0_full.index_copy_(1, dst, y0.index_select(1, src))
    y_seq_full.index_copy_(2, dst, y_seq.index_select(2, src))

    ds_names = lift_info.get("dataset_state_names")
    scaf_names = lift_info.get("scaffold_state_names")
    if ds_names is not None and scaf_names is not None:
        ds_name_to_idx = {str(n): i for i, n in enumerate(ds_names)}
        scaf_obs_set = set(int(i) for i in lift_info["scaffold_obs_idx"])
        for s_idx, s_name in enumerate(scaf_names):
            if s_idx in scaf_obs_set:
                continue
            ds_name = _LIFT_NAME_ALIASES.get(str(s_name), str(s_name))
            if ds_name in ds_name_to_idx:
                y0_full[:, s_idx] = y0[:, ds_name_to_idx[ds_name]]
    return y0_full, y_seq_full


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
            # Fallback: recursive search under datasets/ for the basename or its
            # n1000 variant (sweeps often save a logical name but train on the
            # n=1000 file in a subdir).
            base = Path(cfg["dataset_path"]).name
            stem = Path(base).stem
            for root in (project_root / "datasets", script_dir / "datasets"):
                if not root.exists():
                    continue
                for cand in list(root.rglob(base)) + list(root.rglob(f"{stem}_n1000.npz")):
                    if cand.exists():
                        dataset_path = cand
                        break
                if dataset_path is not None:
                    break
        if dataset_path is None:
            raise FileNotFoundError(
                f"Dataset not found: {cfg['dataset_path']} "
                f"(checked cwd, {exp_dir}, {script_dir}, and {project_root}/datasets)"
            )

    # Match the trainer's filter: if the run used use_synthetic_data=False, the
    # dataset was loaded with synth rows dropped before splitting. Reloading
    # without that flag here would cause the saved train/val/test indices to
    # reference DIFFERENT samples in the unfiltered dataset — endpoint_r2 then
    # skips ~80% of the test set as "synth" because the indices land in the
    # synth region. Forward this flag from the config.
    ds = ODEDataset(dataset_path, use_synthetic_data=bool(cfg.get("use_synthetic_data", True)))

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

    # infer dims (ds.__getitem__ returns (y0, u, y, z_expr) in the current schema)
    item = ds[0]
    y0_ex, u_ex = item[0], item[1]
    P_obs = int(y0_ex.shape[0])
    U = int(u_ex.shape[-1])

    # Partial-observability lift mirror of train.py: when scaffold.P != P_obs,
    # build u_to_y_jump from the scaffold's control_state_map (name-based) and
    # mark the dataset/scaffold obs index pair for later re-packing of y0/y_seq.
    lift_info: dict = {}  # populated when partial-obs lift is active
    if scaffold.P == P_obs:
        jump = make_u_to_y_jump(ds.control_indices, ds.obs_indices, device=device)
    else:
        if (getattr(scaffold, "obs_state_idx", None) is None
                or getattr(scaffold, "control_state_map", None) is None):
            raise ValueError(
                f"Scaffold {scaffold_name} expects P={scaffold.P}, but dataset has P_obs={P_obs}, "
                "and the scaffold did not declare obs_state_idx/control_state_map for the lift."
            )
        if ds.control_names is None:
            raise ValueError("Partial-observability lift needs dataset control_names; rebuild npz.")
        jump = torch.zeros((U, scaffold.P), dtype=torch.float32, device=device)
        for j, name in enumerate(list(ds.control_names)):
            target = scaffold.control_state_map.get(str(name).strip())
            if target is not None:
                jump[j, int(target)] = 1.0
        dataset_obs_idx = list(cfg.get("obs_idx") or scaffold.obs_state_idx)
        if len(dataset_obs_idx) != len(scaffold.obs_state_idx):
            raise ValueError(
                f"cfg.obs_idx (len={len(dataset_obs_idx)}) must match "
                f"scaffold.obs_state_idx (len={len(scaffold.obs_state_idx)})."
            )
        lift_info = {
            "dataset_obs_idx": dataset_obs_idx,
            "scaffold_obs_idx": list(scaffold.obs_state_idx),
            "scaffold_P": int(scaffold.P),
            "dataset_state_names": ([str(n) for n in ds.obs_names]
                                    if getattr(ds, "obs_names", None) is not None else None),
            "scaffold_state_names": (list(scaffold.state_names)
                                     if getattr(scaffold, "state_names", None) is not None else None),
        }
        print(f"[rebuild] partial-obs lift: dataset_obs_idx={dataset_obs_idx} "
              f"→ scaffold_obs_idx={scaffold.obs_state_idx} (P={scaffold.P})")

    # Recompute gru_u_cols / gru_y_cols from config (same logic as train_R.py)
    gru_u_cols = None
    if cfg.get("exclude_ode_cols_from_gru", False):
        gru_u_cols = [j for j in range(U) if int(ds.control_indices[j]) >= scaffold.P]
    elif cfg.get("gru_u_cols") is not None:
        # Explicit gru_u_cols from TrainConfig — must be honored at rebuild
        # time or lift.0.weight shape will not match the checkpoint.
        gru_u_cols = list(cfg["gru_u_cols"])

    gru_y_cols = None
    if lift_info:
        # Partial-obs scaffolds override cfg.gru_y_cols with scaffold.obs_state_idx
        # at train time (so the encoder reads mm/pm at the scaffold's positions).
        # Mirror that here so encoder weights line up with the checkpoint.
        gru_y_cols = list(lift_info["scaffold_obs_idx"])
    elif cfg.get("gru_y_obs_only", False):
        obs_idx_cfg = cfg.get("obs_idx")
        gru_y_cols = list(obs_idx_cfg) if obs_idx_cfg is not None else list(range(scaffold.P))
    elif cfg.get("gru_y_cols") is not None:
        # Explicit gru_y_cols from train.py's TrainConfig — must be honored at
        # rebuild time or lift.0.weight shape will not match the checkpoint.
        gru_y_cols = list(cfg["gru_y_cols"])

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
        # GRU/head variant + init knobs — these change parameter NAMES
        # ("gru.cells.0.weight_ih" vs "gru.weight_ih_l0"), so omitting them
        # makes load_state_dict(strict=False) silently drop the encoder weights
        # and the eval'd model runs with a random GRU. theta_head_tau changes
        # the gamma slope; default 1.0 vs supervisor 2.3 is silently wrong.
        gru_variant=str(cfg.get("gru_variant", "nn_gru")),
        gru_init=str(cfg.get("gru_init", "default")),
        head_init=str(cfg.get("head_init", "default")),
        theta_head_tau=float(cfg.get("theta_head_tau", 1.0)),
        y0_theta_init=bool(cfg.get("y0_theta_init", False)),

        # Sparse-θ knobs. K doesn't affect parameter shapes (only the anchor
        # mask), so checkpoint loading succeeds regardless — but the rollout
        # used to generate plots would fire at the wrong anchors if these were
        # left at defaults. Must mirror train.py's V1/V2 wiring.
        **(
            dict(
                n_theta_anchors=int(cfg["n_theta_anchors"]),
                anchor_interp=str(cfg.get("anchor_interp", "piecewise")),
            )
            if model_class_name in {
                "ode_rnn_sparse_theta",
                "ode_slstm_sparse_theta",
                "ode_rnn_basal_v2_sparse_theta",
            }
            else dict(
                n_theta_anchors=int(cfg["n_theta_anchors"]),
                anchor_mode=str(cfg.get("anchor_mode", "uniform")),
                bolus_max_anchors=int(cfg.get("bolus_max_anchors", 0)),
            )
            if model_class_name == "ode_rnn_sparse_theta_v2"
            else {}
        ),

        # --- NEW ADDITIONS TO SYNC WITH TRAIN.PY ---
        tf_group_size=int(cfg.get("tf_group_size", 32)),
        ar_gap=int(cfg.get("ar_gap", 4)),
        encoder_use_time=bool(cfg.get("encoder_use_time", False)),
        encoder_use_log_dt=bool(cfg.get("encoder_use_log_dt", False)),
        forget_bias_init=cfg.get("forget_bias_init", None),
        legacy_forget_bias_bug=bool(cfg.get("legacy_forget_bias_bug", False)),
        detach_y_prev=bool(cfg.get("detach_y_prev", True)),
        # u_minmax values: try ds attributes first (train_R style). If missing,
        # fall back to reading the raw dataset npz for 'u_scale_max' / 'u_scaled_cols'.
        u_minmax_max=(torch.tensor(getattr(ds, "u_scale_max", None), dtype=torch.float32)
                      if str(cfg.get("u_transform", "none")) in ("minmax", "minmax_sqrt") and getattr(ds, "u_scale_max", None) is not None
                      else None),
        u_minmax_cols=(list(getattr(ds, "u_scaled_cols_idx", None))
                       if str(cfg.get("u_transform", "none")) in ("minmax", "minmax_sqrt") and getattr(ds, "u_scaled_cols_idx", None) is not None
                       else None),
        # If dataset object didn't include minmax info (older ODEDataset), try loading from the npz file.
    ).to(device)

    # Post-init: ensure u_minmax info is properly wired into the model when u_transform requires it.
    # The model uses self.u_minmax_max_full (registered buffer) and self._has_u_minmax.
    # Setting plain attributes like model.u_minmax_max won't help — we must update the buffer.
    if str(cfg.get("u_transform", "none")) in ("minmax", "minmax_sqrt"):
        if not getattr(model, "_has_u_minmax", False):
            try:
                raw = np.load(str(dataset_path), allow_pickle=True)
                u_scale_max_raw = None
                u_scaled_cols_raw = None
                if "u_scale_max" in raw:
                    u_scale_max_raw = torch.tensor(raw["u_scale_max"].astype(np.float32), dtype=torch.float32)
                if "u_scaled_cols" in raw:
                    if getattr(ds, "control_names", None) is not None:
                        scaled = list(raw["u_scaled_cols"].astype(str))
                        ctrl_list = list(ds.control_names)
                        try:
                            u_scaled_cols_raw = [ctrl_list.index(c) for c in scaled]
                        except ValueError:
                            u_scaled_cols_raw = list(raw["u_scaled_cols"].astype(int))
                    else:
                        u_scaled_cols_raw = list(raw["u_scaled_cols"].astype(int))
                if u_scale_max_raw is not None and u_scaled_cols_raw is not None:
                    cols = torch.as_tensor(u_scaled_cols_raw, dtype=torch.long)
                    model.u_minmax_max_full[cols] = u_scale_max_raw.clamp_min(1e-8)
                    model._has_u_minmax = True
            except Exception:
                pass
    # strict=False historically hid silent param-name mismatches (e.g. when
    # gru_variant wasn't propagated to the rebuild and the checkpoint's
    # StackedGRUCellBlock keys silently fell on the floor). Use strict-with-report
    # so any future drift is loud rather than producing a random-encoder model.
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    # Allow non-persistent buffer drift (e.g. gru_u_idx, gru_y_idx, u_minmax_max_full)
    # but flag any param-shaped drift loudly.
    _benign = {"gru_u_idx", "gru_y_idx", "u_minmax_max_full", "theta_lo_vec", "theta_hi_vec", "u_to_y_jump"}
    real_missing = [k for k in missing if k.split(".")[-1] not in _benign]
    real_unexpected = [k for k in unexpected if k.split(".")[-1] not in _benign]
    if real_missing or real_unexpected:
        print(f"[rebuild] WARNING — state_dict mismatch:")
        if real_missing:
            print(f"  missing  ({len(real_missing)}): {real_missing[:8]}{' ...' if len(real_missing) > 8 else ''}")
        if real_unexpected:
            print(f"  unexpect ({len(real_unexpected)}): {real_unexpected[:8]}{' ...' if len(real_unexpected) > 8 else ''}")
        print("  → eval will use RANDOM weights for missing keys. Check rebuild_model_from_experiment kwargs vs train.py.")
    model.eval()

    # state_names: use scaffold-side names when lifting (scaffold layout differs
    # from dataset layout); otherwise prefer dataset obs_names.
    if lift_info and getattr(scaffold, "state_names", None):
        state_names = list(scaffold.state_names)
    else:
        state_names = ds.obs_names.tolist() if ds.obs_names is not None else [f"y{i}" for i in range(P_obs)]
    param_names = scaffold.param_names() if hasattr(scaffold, "param_names") else [f"θ{i}" for i in range(scaffold.theta_dim)]

    return model, ds, state_names, param_names, lift_info


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
    # When `val_n=0`, np.savez(..., val_species_losses=None) round-trips to a 0-d
    # object array containing None. `.size` is 1 (not 0) and `.shape[1]` would raise
    # IndexError. Guard on dtype/ndim before any shape indexing.
    if V.dtype == object or V.ndim < 2 or V.size == 0:
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


def _save_event_driven_dashboard(t, y_true, y_pred, u_val, theta_val, state_names, control_names, param_names, title, out_path):
    """Saves a dashboard grouped by Params and States, with inputs as vertical event lines."""
    P = y_pred.shape[-1]
    U = u_val.shape[-1]
    D = theta_val.shape[-1] if theta_val is not None else 0

    # 1. Identify when and what inputs occur
    events = []
    for j in range(U):
        # Find indices where a bolus is applied (using a small threshold to ignore noise)
        active_indices = np.where(u_val[:, j] > 1e-4)[0]
        for idx in active_indices:
            # Store: (time, input_name, color)
            events.append((t[idx], control_names[j], f'C{j % 10}'))

    # Calculate grid size (4 columns max per row looks clean)
    total_plots = D + P
    cols = 4
    rows = int(np.ceil(total_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows), sharex=True)
    axes = np.array(axes).reshape(-1)
    
    idx = 0

    # 2. Plot Parameters (Theta) First
    if theta_val is not None:
        for d in range(D):
            ax = axes[idx]
            ax.plot(t, theta_val[:, d], linewidth=2, color='C3')
            name = param_names[d] if (param_names is not None and d < len(param_names)) else f"θ{d}"
            ax.set_title(f"Param: {name}", fontsize=11, fontweight='bold', color='C3')
            
            # Draw event lines
            for e_time, e_name, e_color in events:
                ax.axvline(e_time, color=e_color, linestyle='--', alpha=0.6, linewidth=1.5)
            
            ax.grid(True, alpha=0.3)
            ax.set_xlim([t[0], t[-1]])
            idx += 1

    # 3. Plot States (Trajectories) Next
    for p in range(P):
        ax = axes[idx]
        ax.plot(t, y_true[:, p], linewidth=2, label="true", color='C0')
        ax.plot(t, y_pred[:, p], linewidth=2, linestyle="--", label="pred", color='C1')
        ax.set_title(f"State: {state_names[p]}", fontsize=11, fontweight='bold')
        
        # Draw event lines
        for e_time, e_name, e_color in events:
            ax.axvline(e_time, color=e_color, linestyle='--', alpha=0.6, linewidth=1.5)
            
        ax.grid(True, alpha=0.3)
        ax.set_xlim([t[0], t[-1]])
        if p == 0:
            ax.legend(fontsize=9, loc='upper right')
        idx += 1

    # Hide unused subplots
    for i in range(idx, len(axes)):
        axes[i].axis("off")

    # Add x-labels to the bottom-most visible plots
    for ax in axes:
        if ax.get_subplotspec().is_last_row() or ax.get_subplotspec().rowspan.stop == rows:
            ax.set_xlabel("Time", fontsize=10)

    # 4. Title on top, bolus-event legend just below it (so they don't overlap)
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=f'C{j % 10}', linestyle='--', lw=2) for j in range(U)]
    fig.suptitle(title, fontsize=16, y=0.985)
    fig.legend(
        custom_lines, control_names,
        loc='upper center', bbox_to_anchor=(0.5, 0.945),
        ncol=U, title="Bolus Events", fontsize=10, title_fontsize=12,
    )
    # Reserve the top ~12% of the figure for title + legend.
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_predictions(model, ds: ODEDataset, state_names: list[str], out_dir: Path, n_samples: int, device: torch.device, exp_dir: Path | None = None, u_transform: str = "none", param_names: list[str] | None = None, lift_info: dict | None = None, extra_new_samples: int = 0):
    raw_ds = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    plot_ds = _test_subset(ds, exp_dir) if exp_dir is not None else ds
    n_samples = min(int(n_samples), len(plot_ds))
    collate_fn = collate_varlen if getattr(raw_ds, "variable_length", False) else collate
    split_label = "test" if (exp_dir is not None and (exp_dir / "split.npz").exists()) else "train"

    if isinstance(plot_ds, torch.utils.data.Subset):
        plotted_indices = list(plot_ds.indices[:n_samples])
    else:
        plotted_indices = list(range(n_samples))

    loader = torch.utils.data.DataLoader(plot_ds, batch_size=n_samples, shuffle=False, num_workers=0, collate_fn=collate_fn)
    batch = next(iter(loader))
    # collate / collate_varlen now return (y0, u, y, lengths_or_None, z_expr, dt_seq)
    y0, u_seq, y_seq, batch_lengths = batch[0], batch[1], batch[2], batch[3]
    dt_seq = batch[5] if len(batch) >= 6 else torch.from_numpy(raw_ds.dt[:u_seq.shape[1]])[None, :].expand(y0.shape[0], -1)

    y0 = y0.to(device)
    u_seq = u_seq.to(device)
    y_seq = y_seq.to(device)
    dt_seq = dt_seq.to(device)
    if batch_lengths is not None:
        batch_lengths = batch_lengths.to(device)
    # Lift dataset y0/y_seq into scaffold-state space when partial-obs.
    y0, y_seq = _maybe_lift(y0, y_seq, lift_info or {})

    # Diagnostic info before calling model
    print(f"[plot_predictions] n_samples={n_samples}, plot_ds_len={len(plot_ds)}")
    print(f"[plot_predictions] y0.shape={tuple(y0.shape)}, u_seq.shape={tuple(u_seq.shape)}, y_seq.shape={tuple(y_seq.shape)}, dt_seq.shape={tuple(dt_seq.shape)}")
    with torch.no_grad():
        # When lifting, model emits in scaffold layout; obs_idx points to the
        # scaffold's obs positions so downstream slicing extracts the right cols.
        if lift_info:
            obs_idx = torch.tensor(lift_info["scaffold_obs_idx"], device=y0.device, dtype=torch.long)
        else:
            obs_idx = torch.arange(y0.shape[-1], device=y0.device)
        cfg_local = load_yaml(exp_dir / "config.yaml") if exp_dir else {}
        y_transform = str(cfg_local.get("y_transform", "none"))
        model_kwargs = {"y_seq": None, "teacher_forcing": False, "u_transform": u_transform, "y_transform": y_transform}
        if model.__class__.__name__ == "OdeTransformerGrouped" and batch_lengths is not None:
            model_kwargs["lengths"] = batch_lengths
        try:
            pred, _theta, _beta = model(y0, u_seq, dt_seq, obs_idx, **_filter_model_kwargs(model, model_kwargs))
        except Exception as e:
            import traceback
            print(f"[plot_predictions] ERROR calling model.forward: {e}")
            traceback.print_exc()
            return

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
        # Plot 4: Event-Driven Dashboard
        # ---------------------------------------------------------
        _save_event_driven_dashboard(
            t, y_true_i, y_pred_i, u_val_i, theta_val_i, 
            state_names, control_names, param_names, 
            title=f"Event Overview [{split_label}] (sample {sample_id_str})", 
            out_path=out_dir / f"event_dashboard_{sample_id_str}.png"
        )

    if theta_np is None:
        print("[warn] model returned no theta; skipping theta plots for plotted samples")

    # ─── extra plots from NEW samples only ───────────────────────────────────
    # When the dataset has a source_label field (i.e. it's a mixed OLD+NEW
    # native-grid build) and extra_new_samples > 0, plot that many additional
    # NEW samples from the test split with a "new_" filename prefix. Lets you
    # eyeball post-tube-opening behaviour without manually digging out indices.
    if extra_new_samples > 0:
        src = getattr(raw_ds, "source_label", None)
        if src is None:
            print(f"[plot_predictions] extra_new_samples={extra_new_samples} requested but dataset has no source_label; skipping")
        else:
            src_arr = np.array([str(s) for s in src])
            test_idx_arr = np.array(plotted_indices) if isinstance(plot_ds, torch.utils.data.Subset) else np.arange(len(plot_ds))
            # Use the full test split, not just the already-plotted slice
            if isinstance(plot_ds, torch.utils.data.Subset):
                full_test_idx = np.asarray(plot_ds.indices, dtype=int)
            else:
                full_test_idx = np.arange(len(plot_ds))
            new_in_test = full_test_idx[src_arr[full_test_idx] == "new"]
            if len(new_in_test) == 0:
                print(f"[plot_predictions] no NEW samples in test split; skipping {extra_new_samples} extras")
            else:
                chosen = list(new_in_test[: int(extra_new_samples)])
                print(f"[plot_predictions] plotting {len(chosen)} extra NEW samples: idx={chosen}")
                sub = torch.utils.data.Subset(raw_ds, chosen)
                ldr = torch.utils.data.DataLoader(sub, batch_size=len(chosen), shuffle=False, num_workers=0, collate_fn=collate_fn)
                bx = next(iter(ldr))
                y0x, u_seqx, y_seqx, blx = bx[0], bx[1], bx[2], bx[3]
                dt_seqx = bx[5] if len(bx) >= 6 else torch.from_numpy(raw_ds.dt[:u_seqx.shape[1]])[None, :].expand(y0x.shape[0], -1)
                y0x, u_seqx, y_seqx, dt_seqx = y0x.to(device), u_seqx.to(device), y_seqx.to(device), dt_seqx.to(device)
                if blx is not None:
                    blx = blx.to(device)
                y0x, y_seqx = _maybe_lift(y0x, y_seqx, lift_info or {})
                with torch.no_grad():
                    obs_idx_x = (torch.tensor(lift_info["scaffold_obs_idx"], device=y0x.device, dtype=torch.long)
                                 if lift_info else torch.arange(y0x.shape[-1], device=y0x.device))
                    kwargs_x = {"y_seq": None, "teacher_forcing": False, "u_transform": u_transform, "y_transform": str((load_yaml(exp_dir / "config.yaml") if exp_dir else {}).get("y_transform", "none"))}
                    if model.__class__.__name__ == "OdeTransformerGrouped" and blx is not None:
                        kwargs_x["lengths"] = blx
                    predx, thetax, _ = model(y0x, u_seqx, dt_seqx, obs_idx_x, **_filter_model_kwargs(model, kwargs_x))
                yt = y_seqx.cpu().numpy(); yp = predx.cpu().numpy()
                un = u_seqx.cpu().numpy(); dtn = dt_seqx.cpu().numpy()
                ln = blx.cpu().numpy() if blx is not None else None
                thn = thetax.cpu().numpy() if thetax is not None else None
                Px = yp.shape[-1]
                for i, gi in enumerate(chosen):
                    Li = int(ln[i]) if ln is not None else yp.shape[1]
                    tg = np.concatenate([[0.0], np.cumsum(dtn[i, :Li])])[1:]
                    sid = f"new_{i:03d}_idx{gi}"
                    yt_i, yp_i, u_i = yt[i, :Li, :], yp[i, :Li, :], un[i, :Li, :]
                    th_i = thn[i, :Li, :] if thn is not None else None
                    fig, axes = plt.subplots(Px, 1, figsize=(6, 4 * Px), sharex=True)
                    if Px == 1:
                        axes = [axes]
                    for p, ax in enumerate(axes):
                        ax.plot(tg, yt_i[:, p], linewidth=2, label="true")
                        ax.plot(tg, yp_i[:, p], linewidth=2, linestyle="--", label="pred")
                        ax.set_ylabel(state_names[p] if p < len(state_names) else f"s{p}")
                        ax.grid(True, alpha=0.25)
                        if p == 0:
                            ax.legend()
                    axes[-1].set_xlabel("Time")
                    fig.suptitle(f"Prediction vs truth [NEW test sample {sid}]")
                    fig.tight_layout()
                    fig.savefig(out_dir / f"pred_vs_true_{sid}.png", dpi=150)
                    plt.close(fig)
                    _save_bolus_panel(tg, u_i, control_names,
                                      title=f"Inputs/Boluses [NEW test sample {sid}]",
                                      out_path=out_dir / f"inputs_bolus_{sid}.png")
                    if th_i is not None:
                        _save_theta_panel(th_i, dtn[i, :Li],
                                          out_dir / f"theta_sample_{sid}.png",
                                          title=f"Learned θ(t) [NEW test sample {sid}]",
                                          param_names=param_names)
                    _save_event_driven_dashboard(tg, yt_i, yp_i, u_i, th_i,
                                                 state_names, control_names, param_names,
                                                 title=f"Event Overview [NEW test sample {sid}]",
                                                 out_path=out_dir / f"event_dashboard_{sid}.png")


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


def plot_beta(model, ds: ODEDataset, state_names: list[str], out_dir: Path, sample_idx: int, device: torch.device, exp_dir: Path | None = None, u_transform: str = "none", lift_info: dict | None = None):
    """Plot the learned beta(t) residual terms."""
    sample_idx = int(sample_idx)
    raw_ds = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    plot_ds = _test_subset(ds, exp_dir) if exp_dir is not None else ds
    item = plot_ds[sample_idx]
    y0, u_seq = item[0], item[1]
    # __getitem__ returns dt at index 3 (after y0, u, y) for both varlen and fixed.
    dt_seq = item[3] if len(item) >= 4 and torch.is_tensor(item[3]) else torch.from_numpy(raw_ds.dt[:int(u_seq.shape[0])])

    y0 = y0.unsqueeze(0).to(device)
    u_seq = u_seq.unsqueeze(0).to(device)
    dt_seq = dt_seq.unsqueeze(0).to(device)
    if lift_info:
        y0_dummy_y = torch.zeros(1, u_seq.shape[1], y0.shape[-1], device=device)
        y0, _ = _maybe_lift(y0, y0_dummy_y, lift_info)

    print(f"[plot_beta] sample_idx={sample_idx}, y0.shape={tuple(y0.shape)}, u_seq.shape={tuple(u_seq.shape)}, dt_seq.shape={tuple(dt_seq.shape)}")
    with torch.no_grad():
        if lift_info:
            obs_idx = torch.tensor(lift_info["scaffold_obs_idx"], device=y0.device, dtype=torch.long)
        else:
            obs_idx = torch.arange(y0.shape[-1], device=y0.device)
        cfg_local = load_yaml(exp_dir / "config.yaml") if exp_dir else {}
        y_transform = str(cfg_local.get("y_transform", "none"))
        model_kwargs = {"y_seq": None, "teacher_forcing": False, "u_transform": u_transform, "y_transform": y_transform}
        if model.__class__.__name__ == "OdeTransformerGrouped":
            model_kwargs["lengths"] = torch.tensor([K], device=y0.device, dtype=torch.long)
        try:
            _, _theta, beta = model(y0, u_seq, dt_seq, obs_idx, **_filter_model_kwargs(model, model_kwargs))
        except Exception as e:
            import traceback
            print(f"[plot_beta] ERROR calling model.forward: {e}")
            traceback.print_exc()
            return

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


def plot_experiment(exp_dir: str | Path, n_samples: int = 5, sample_idx: int = 0, ckpt_path: str | Path | None = None, extra_new_samples: int = 0) -> Path:
    exp_dir = Path(exp_dir)
    out_dir = exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device_auto()
    model, ds, state_names, param_names, lift_info = rebuild_model_from_experiment(exp_dir, device=device, ckpt_path=ckpt_path)

    cfg = load_yaml(exp_dir / "config.yaml")
    # cfg.get("obs_idx") can be None (explicit null in YAML). Coerce safely.
    # For partial-obs runs, the val_species heatmap was logged in scaffold layout
    # at the scaffold's obs positions; map labels accordingly.
    if lift_info:
        obs_idx = list(lift_info["scaffold_obs_idx"])
    else:
        obs_idx = list(cfg.get("obs_idx") or []) or None
    u_transform = str(cfg.get("u_transform", "none"))

    loss_npz = exp_dir / "logs" / "loss_curves.npz"
    if loss_npz.exists():
        plot_loss_curves(loss_npz, out_dir)
        plot_val_species_losses(loss_npz, out_dir, state_names=state_names, obs_idx=obs_idx)

    plot_predictions(model, ds, state_names=state_names, out_dir=out_dir, n_samples=n_samples, device=device, exp_dir=exp_dir, u_transform=u_transform, param_names=param_names, lift_info=lift_info, extra_new_samples=int(extra_new_samples))

    if cfg.get("use_basal", False) or cfg.get("model_class", "") == "ode_rnn_basal_v2":
        plot_beta(model, ds, state_names=state_names, out_dir=out_dir, sample_idx=sample_idx, device=device, exp_dir=exp_dir, u_transform=u_transform, lift_info=lift_info)

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

    model, ds, state_names, _param_names, lift_info = rebuild_model_from_experiment(exp_dir, device=device)
    cfg_overlay = load_yaml(exp_dir / "config.yaml")
    u_transform_overlay = str(cfg_overlay.get("u_transform", "none"))
    y_transform_overlay = str(cfg_overlay.get("y_transform", "none"))

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
    item = ds[sample_idx]
    y0, u_seq, y_seq = item[0], item[1], item[2]
    K = int(u_seq.shape[0])
    # Use per-sample dt from __getitem__ (index 3) when available — falls back
    # to the dataset's shared dt for back-compat with older NPZs.
    dt_t = item[3] if len(item) >= 4 and torch.is_tensor(item[3]) else torch.from_numpy(raw_ds.dt[:K])
    dt = dt_t.detach().cpu().numpy().astype(np.float32)
    t = np.cumsum(dt)

    y0_b = y0.unsqueeze(0).to(device)
    u_b = u_seq.unsqueeze(0).to(device)
    dt_b = dt_t.unsqueeze(0).to(device)
    y_seq_b = y_seq.unsqueeze(0).to(device)
    # Lift dataset (y0, y_seq) into scaffold layout for partial-obs runs.
    y0_b, y_seq_b = _maybe_lift(y0_b, y_seq_b, lift_info or {})

    y_true = y_seq_b[0].cpu().numpy()
    P = int(y_true.shape[1])

    preds: dict[int, np.ndarray] = {}
    for ep in chosen:
        ckpt_path = epoch_to_path[ep]
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" not in ckpt:
            raise RuntimeError(f"Checkpoint missing 'state_dict': {ckpt_path}")

        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()
        with torch.no_grad():
            if lift_info:
                obs_idx = torch.tensor(lift_info["scaffold_obs_idx"], device=y0_b.device, dtype=torch.long)
            else:
                obs_idx = torch.arange(y0_b.shape[-1], device=y0_b.device)
            model_kwargs = {"y_seq": None, "teacher_forcing": False, "u_transform": u_transform_overlay, "y_transform": y_transform_overlay}
            if model.__class__.__name__ == "OdeTransformerGrouped":
                model_kwargs["lengths"] = torch.tensor([K], device=y0_b.device, dtype=torch.long)
            try:
                pred, _theta, _beta = model(y0_b, u_b, dt_b, obs_idx, **_filter_model_kwargs(model, model_kwargs))
            except Exception as e:
                import traceback
                print(f"[epoch overlay] ERROR calling model.forward for epoch {ep}: {e}")
                traceback.print_exc()
                raise
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