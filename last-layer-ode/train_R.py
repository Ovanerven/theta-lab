# train.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional
import math
from pathlib import Path
import re
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

from scaffolds import SCAFFOLDS
from models import MODELS
from jumps import make_u_to_y_jump


class ODEDataset(Dataset):
    """
    Required npz fields:
      y0    : (N,P_obs)
      u_seq : (N,K,U)
      y_seq : (N,K,P_obs)
      t_obs : (K+1,)
      control_indices : (U,)
      obs_indices     : (P_obs,)
    Optional:
      names_full, control_names, obs_names
      lengths : (N,) — per-sample valid sequence length.
                When present, __getitem__ returns trimmed tensors
                and collate_varlen pads each batch to its own max.
    """

    def __init__(self, npz_path: str | Path):
        d = np.load(str(npz_path), allow_pickle=True)

        self.y0 = d["y0"].astype(np.float32)  # (N,P_obs)
        self.u_seq = d["u_seq"].astype(np.float32)  # (N,K,U)
        self.y_seq = d["y_seq"].astype(np.float32)  # (N,K,P_obs)
        t_obs = d["t_obs"].astype(np.float32)  # (K+1,)
        self.dt = np.diff(t_obs).astype(np.float32)  # (K,)

        if "control_indices" not in d or "obs_indices" not in d:
            raise ValueError(
                f"Dataset {npz_path} missing control_indices/obs_indices. Regenerate dataset with metadata."
            )
        self.control_indices = d["control_indices"].astype(np.int64)
        self.obs_indices = d["obs_indices"].astype(np.int64)

        self.names_full = d["names_full"].astype(str) if "names_full" in d else None
        self.control_names = d["control_names"].astype(str) if "control_names" in d else None
        self.obs_names = d["obs_names"].astype(str) if "obs_names" in d else None

        # Variable-length support
        if "lengths" in d:
            self.lengths = d["lengths"].astype(np.int64)  # (N,)
            self.variable_length = True
        else:
            self.lengths = None
            self.variable_length = False

    def __len__(self) -> int:
        return self.y0.shape[0]

    def __getitem__(self, i: int):
        if self.variable_length:
            L = int(self.lengths[i])
            return (
                torch.from_numpy(self.y0[i]),          # (P_obs,)
                torch.from_numpy(self.u_seq[i, :L]),   # (L,U)
                torch.from_numpy(self.y_seq[i, :L]),   # (L,P_obs)
            )
        return (
            torch.from_numpy(self.y0[i]),  # (P_obs,)
            torch.from_numpy(self.u_seq[i]),  # (K,U)
            torch.from_numpy(self.y_seq[i]),  # (K,P_obs)
        )


def collate(batch):
    y0, u, y = zip(*batch)
    return torch.stack(y0), torch.stack(u), torch.stack(y), None


def collate_varlen(batch):
    """Pad each batch to its own max length; return lengths tensor."""
    y0_list, u_list, y_list = zip(*batch)
    lengths = torch.tensor([u.shape[0] for u in u_list], dtype=torch.long)
    y0 = torch.stack(y0_list)
    u_padded = torch.nn.utils.rnn.pad_sequence(u_list, batch_first=True)   # (B, K_batch, U)
    y_padded = torch.nn.utils.rnn.pad_sequence(y_list, batch_first=True)   # (B, K_batch, P)
    return y0, u_padded, y_padded, lengths


def _build_loss_mask(lengths: torch.Tensor, K: int, device: torch.device) -> torch.Tensor:
    """Build (B, K) boolean mask: True for valid timesteps."""
    return torch.arange(K, device=device).unsqueeze(0) < lengths.unsqueeze(1)


def loss_fn(
    pred: torch.Tensor,
    y_seq: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    loss_type: str = "log_mse",
    species_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-species loss, optionally masking padded timesteps.

    loss_type:
      - 'log_mse': MSE in log1p space (default, legacy behaviour).
      - 'mse'    : MSE in raw space.
      - 'rmse'   : sqrt(MSE) in raw space.

    species_weights: (P,) tensor applied multiplicatively on the per-species
    squared error before reduction. Typically set to 1 / mean_per_species so
    species with different dynamic ranges contribute comparably.
    """
    if loss_type == "log_mse":
        se = (torch.log1p(pred) - torch.log1p(y_seq)).pow(2)
    elif loss_type in ("mse", "rmse"):
        se = (pred - y_seq).pow(2)
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}; expected 'log_mse', 'mse', or 'rmse'.")

    if species_weights is not None:
        se = se * species_weights.view(1, 1, -1)

    if lengths is not None:
        mask = _build_loss_mask(lengths, se.shape[1], se.device)  # (B,K)
        se = se * mask.unsqueeze(-1)
        denom = mask.sum() * se.shape[-1]
        out = se.sum() / denom
    else:
        out = se.mean()

    if loss_type == "rmse":
        out = torch.sqrt(out + 1e-12)
    return out


def _build_tail_mask(
    lengths: Optional[torch.Tensor],
    K: int,
    device: torch.device,
    frac: float,
    batch_size: int,
) -> torch.Tensor:
    """Boolean (B, K) mask selecting the final `frac` of each sample's valid length."""
    if lengths is None:
        lens = torch.full((batch_size,), K, device=device, dtype=torch.long)
    else:
        lens = lengths
    tail = torch.clamp((lens.float() * frac).long(), min=1)
    idx = torch.arange(K, device=device).unsqueeze(0)  # (1,K)
    start = (lens - tail).unsqueeze(1)                 # (B,1)
    end = lens.unsqueeze(1)                            # (B,1)
    return (idx >= start) & (idx < end)


def r_terminal_loss(
    pred_full: torch.Tensor,
    r_idx: int,
    lengths: Optional[torch.Tensor],
    frac: float = 0.05,
) -> torch.Tensor:
    """Force R -> 0 over the final `frac` of each trajectory (target column is zeros)."""
    B, K = pred_full.shape[0], pred_full.shape[1]
    r = pred_full[:, :, r_idx]  # (B,K)
    mask = _build_tail_mask(lengths, K, pred_full.device, frac, B).to(r.dtype)
    se = (r ** 2) * mask
    denom = mask.sum().clamp(min=1.0)
    return se.sum() / denom


def loss_fn_per_species(
    pred: torch.Tensor,
    y_seq: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    loss_type: str = "log_mse",
    species_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if loss_type == "log_mse":
        se = (torch.log1p(pred) - torch.log1p(y_seq)).pow(2)
    elif loss_type in ("mse", "rmse"):
        se = (pred - y_seq).pow(2)
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    if species_weights is not None:
        se = se * species_weights.view(1, 1, -1)

    if lengths is not None:
        mask = _build_loss_mask(lengths, se.shape[1], se.device)
        se = se * mask.unsqueeze(-1)
        out = se.sum(dim=(0, 1)) / mask.sum()
    else:
        out = se.mean(dim=(0, 1))

    if loss_type == "rmse":
        out = torch.sqrt(out + 1e-12)
    return out


def compute_species_mean_weights(
    y_seq: np.ndarray,
    lengths: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Per-species weight = 1 / mean_over_valid_timesteps(y_seq).

    Returns a (P,) float32 vector. Intended for normalizing species that
    differ in dynamic range (e.g. mRNA vs protein) so each contributes
    comparably to the loss.
    """
    y = np.asarray(y_seq, dtype=np.float64)  # (N,K,P)
    N, K, P = y.shape
    if lengths is None:
        means = y.reshape(-1, P).mean(axis=0)
    else:
        lens = np.asarray(lengths, dtype=np.int64)
        mask = (np.arange(K)[None, :] < lens[:, None]).astype(np.float64)  # (N,K)
        num = (y * mask[..., None]).sum(axis=(0, 1))                        # (P,)
        den = mask.sum() + eps
        means = num / den
    return (1.0 / np.maximum(means, eps)).astype(np.float32)


@dataclass
class TrainConfig:
    dataset_path: str

    study: str = "adhoc"
    tags: list[str] | None = None
    exp_name: str = "default"
    out_root: str = "experiments"

    save_model_name: str = "model.pt"  # saved in exp_dir/
    save_last_name: str = "model_last.pt"  # saved in exp_dir/
    save_curves_name: str = "loss_curves.npz"  # saved in exp_dir/logs/

    epochs: int = 200
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 0  # linear LR warmup; 0 disables
    cosine_decay: bool = False  # cosine decay from lr to lr*cosine_decay_min after warmup
    val_n: int = 100   # fixed count for validation set
    test_n: int = 100  # fixed count for held-out test set
    # legacy: val_frac still accepted but val_n/test_n take precedence when > 0
    val_frac: float = 0.0
    seed: int = 42        # controls model init + training stochasticity only
    split_seed: int = 42   # controls train/val/test split — keep fixed across seeds

    num_workers: int = 0
    pin_memory: bool = True

    scaffold: str = "reduced5"
    hidden: int = 128
    lift_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    theta_lo: float = 1e-3
    theta_hi: float = 2.0
    n_substeps: int = 1

    ff_mult: int = 2  # feedforward multiplier inside each transformer layer (2=current, 4=standard)

    # Transformer-specific (ignored by non-Transformer models via **kwargs)
    context_len: int = 64  # sliding window size; set >= sequence length for full context
    tf_group_size: int = 32  # grouped-TF chunk size for ode_transformer_grouped
    ar_gap: int = 4  # autoregressive steps inserted between grouped-TF chunks

    # Mamba-specific (ignored by non-Mamba models via **kwargs)
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4

    forget_bias_init: Optional[float] = None  # None = PyTorch default; 1.0 = Gers/Jozefowicz positive shift
    legacy_forget_bias_bug: bool = False      # reproduce pre-fix fill_(0.0) on both bias_ih and bias_hh

    use_basal: bool = False
    beta_regularization: bool = False
    lambda_beta: float = 1.0

    theta_bounded: bool = True   # if False, use softplus (unbounded above) instead of gamma

    grad_clip: float = 1.0
    teacher_forcing: bool = True
    tf_every: int = 50
    tf_drop_epoch: int = 10**9

    # checkpointing cadence (0 disables periodic ckpts)
    ckpt_every: int = 10

    l1_regularization: bool = False   # smoothness: penalizes mean |theta[t] - theta[t-1]|
    l2_regularization: bool = False   # smoothness: penalizes mean (theta[t] - theta[t-1])^2

    lambda_reg: float = 0.001

    # Loss type: 'log_mse' (default, legacy), 'mse', 'rmse'.
    loss_type: str = "log_mse"
    # If True, weight per-species SE by 1/mean(species) computed from the
    # training set (real data scenario). Keeps mRNA/protein comparable.
    normalize_by_species_mean: bool = False

    # Terminal R -> 0 regularization (TXTLResourceandMaturationDNAScaffold).
    # Adds loss pushing the R state to 0 over the final `r_terminal_frac` of
    # each trajectory. Disabled when lambda_r_terminal == 0.
    lambda_r_terminal: float = 0.0
    r_terminal_frac: float = 0.05

    # If set (e.g. [0, 12]), supervise loss/TF only on those species indices.
    # If null/None, supervises all observed species (default behaviour).
    obs_idx: list[int] | None = None

    wandb_enabled: bool = False
    wandb_project: str = "theta-lab"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str = "train"
    wandb_name: str | None = None
    wandb_mode: str | None = None

    jit_scripting: bool = False
    torch_compile: bool = False
    autocast_bf16: bool = False

    endpoint_r2: bool = False  # if True, runs endpoint R² analysis & saves plot in exp_dir

    # 'ode_rnn' (default), 'ode_rnn_2020' (latent ODE-RNN style),
    # or 'neural_ode' (pure MLP baseline)
    model_class: str = "ode_rnn"


def load_cfg(path: str | Path) -> TrainConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return TrainConfig(**d)



def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    text = text.strip("-_.")
    return text or "run"


def build_run_dir(cfg: TrainConfig, now: datetime) -> tuple[Path, str, str]:
    study_slug = slugify(cfg.study)
    run_name = slugify(cfg.exp_name)
    run_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{run_name}"
    run_dir = Path(cfg.out_root) / study_slug / run_id
    return run_dir, run_id, study_slug


def init_wandb(cfg: TrainConfig, cfg_dict: dict, *, run_id: str, exp_dir: Path):
    if not cfg.wandb_enabled:
        return None, None

    try:
        import wandb
    except ImportError:
        print("[wandb] wandb is not installed; continuing without W&B logging.")
        return None, None

    raw_tags = cfg.tags or []
    if isinstance(raw_tags, str):
        tags = [raw_tags]
    else:
        tags = [str(tag) for tag in raw_tags]
    if cfg.study not in tags:
        tags.append(str(cfg.study))

    init_kwargs = {
        "project": cfg.wandb_project,
        "entity": cfg.wandb_entity,
        "group": cfg.wandb_group or cfg.study,
        "job_type": cfg.wandb_job_type,
        "name": cfg.wandb_name or run_id,
        "tags": tags,
        "config": cfg_dict,
        "dir": str(exp_dir),
    }
    if cfg.wandb_mode is not None:
        init_kwargs["mode"] = cfg.wandb_mode

    try:
        run = wandb.init(**{k: v for k, v in init_kwargs.items() if v is not None})
    except Exception as exc:
        print(f"[wandb] init failed: {exc}")
        return None, None

    if run is not None:
        run.config.update(
            {
                "run_id": run_id,
                "study": cfg.study,
                "run_dir": str(exp_dir.resolve()),
            },
            allow_val_change=True,
        )
    return wandb, run


def log_wandb_images(wandb, run, plots_dir: Path) -> None:
    if wandb is None or run is None or not plots_dir.exists():
        return

    single_images = [
        ("plots/loss_curves", plots_dir / "loss_curves.png"),
        ("plots/val_species_heatmap", plots_dir / "val_species_heatmap.png"),
        ("plots/val_species_final", plots_dir / "val_species_final.png"),
        ("plots/pred_overlays", plots_dir / "pred_overlays_sample000.png"),
        ("plots/theta_sample0", plots_dir / "theta_sample0.png"),
    ]

    payload = {}
    for key, path in single_images:
        if path.exists():
            payload[key] = wandb.Image(str(path))

    pred_paths = sorted(plots_dir.glob("pred_vs_true_*.png"))[:3]
    if pred_paths:
        payload["plots/pred_vs_true_examples"] = [
            wandb.Image(str(path), caption=path.stem) for path in pred_paths
        ]

    if payload:
        run.log(payload)


def log_wandb_artifact(wandb, run, *, exp_dir: Path, run_id: str) -> None:
    if wandb is None or run is None:
        return

    artifact = wandb.Artifact(run_id, type="experiment")
    for rel_path in [
        Path("config.yaml"),
        Path("model.pt"),
        Path("model_last.pt"),
        Path("logs") / "loss_curves.npz",
    ]:
        path = exp_dir / rel_path
        if path.exists():
            artifact.add_file(str(path), name=str(rel_path))

    try:
        run.log_artifact(artifact)
    except Exception as exc:
        print(f"[wandb] artifact logging failed: {exc}")


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(cfg: TrainConfig, *, no_plot: bool = False, plot_samples: int = 5, plot_sample_idx: int = 0) -> None:
    t0 = time.time()

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = device_auto()
    print(f"Using device: {device}")

    now = datetime.now()
    exp_dir, run_id, study_slug = build_run_dir(cfg, now)
    exp_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = exp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config for later reconstruction
    cfg_dict = asdict(cfg)
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    print(f"Experiment: {exp_dir}")
    print(f"Study: {study_slug} | Run ID: {run_id}")

    wandb, wandb_run = init_wandb(cfg, cfg_dict, run_id=run_id, exp_dir=exp_dir)

    ds = ODEDataset(cfg.dataset_path)
    N = len(ds)

    idx = np.arange(N)
    rng = np.random.default_rng(cfg.split_seed)
    rng.shuffle(idx)

    # fixed-count split: test / val / train
    n_test = int(cfg.test_n) if cfg.test_n > 0 else 0
    n_val  = int(cfg.val_n)  if cfg.val_n  > 0 else max(1, int(N * cfg.val_frac))
    if n_test + n_val >= N:
        raise ValueError(f"val_n={n_val} + test_n={n_test} >= N={N}")
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    # persist split so plotting always uses the correct test indices
    np.savez(exp_dir / "split.npz",
             train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    collate_fn = collate_varlen if ds.variable_length else collate

    train_loader = DataLoader(
        torch.utils.data.Subset(ds, train_idx.tolist()),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=bool(cfg.pin_memory),
    )

    val_loader = None
    if len(val_idx) > 0:
        val_loader = DataLoader(
            torch.utils.data.Subset(ds, val_idx.tolist()),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=bool(cfg.pin_memory),
        )

    # infer dims
    y0_ex, u_ex, _ = ds[0]
    P_obs = int(y0_ex.shape[0])
    U = int(u_ex.shape[-1])

    if cfg.scaffold not in SCAFFOLDS:
        raise ValueError(f"Unknown scaffold '{cfg.scaffold}'. Available: {list(SCAFFOLDS.keys())}")
    scaffold = SCAFFOLDS[cfg.scaffold]

    if scaffold.P != P_obs:
        raise ValueError(f"Scaffold {cfg.scaffold} expects P={scaffold.P}, but dataset has P_obs={P_obs}.")

    u_to_y_jump = make_u_to_y_jump(ds.control_indices, ds.obs_indices, device=device)  # (U,P_obs)

    if cfg.model_class not in MODELS:
        raise ValueError(f"Unknown model_class '{cfg.model_class}'. Available: {list(MODELS.keys())}")
    model = MODELS[cfg.model_class](
        U=U,
        rhs=scaffold,
        u_to_y_jump=u_to_y_jump,
        hidden=cfg.hidden,
        lift_dim=cfg.lift_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        ff_mult=cfg.ff_mult,
        theta_lo=cfg.theta_lo,
        theta_hi=cfg.theta_hi,
        n_substeps=cfg.n_substeps,
        use_basal=cfg.use_basal,
        context_len=cfg.context_len,
        tf_group_size=cfg.tf_group_size,
        ar_gap=cfg.ar_gap,
        theta_bounded=cfg.theta_bounded,
        d_state=cfg.d_state,
        expand=cfg.expand,
        d_conv=cfg.d_conv,
        forget_bias_init=cfg.forget_bias_init,
        legacy_forget_bias_bug=cfg.legacy_forget_bias_bug,
    ).to(device)

    # DIAGNOSTIC ONLY — safe to remove once confirmed Flash Attention works on your GPU.
    if cfg.autocast_bf16 and cfg.model_class in {"ode_transformer", "ode_transformer_grouped"} and device.type == "cuda":
        _B, _W, _H = 2, 8, model.hidden
        _x = torch.randn(_B, _W, _H, device=device, dtype=torch.bfloat16)
        _mask = nn.Transformer.generate_square_subsequent_mask(_W, device=device).to(torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                try:
                    model.transformer(_x, mask=_mask, is_causal=True)
                    print("Flash Attention: OK")
                except Exception as e:
                    print(f"Flash Attention: NOT available ({e})")
        del _x, _mask

    compile_model = cfg.torch_compile
    jit_scripting = cfg.jit_scripting

    if jit_scripting == True:
        try:
            model = torch.jit.script(model)
            print('The model compiled successfully')
        except:
            print('The model did not compile please check')

    elif compile_model == True:
        try:
            # dynamic=True: generates one symbolic kernel per distinct graph instead of
            # recompiling for every new tensor shape. Critical for the OdeTransformer
            # whose attention window W grows 1..context_len during the forward loop —
            # without this, Dynamo recompiles 64 times (~hours on GPFS clusters).
            model = torch.compile(model, dynamic=True)
            print("torch.compile: OK (dynamic=True)")
        except Exception as e:
            print(f'The model did not compile: {e}')


    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    scheduler = None
    if cfg.warmup_epochs > 0 and cfg.cosine_decay:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-6, end_factor=1.0, total_iters=int(cfg.warmup_epochs)
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, cfg.epochs - cfg.warmup_epochs), eta_min=float(cfg.lr) * 0.01
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[int(cfg.warmup_epochs)]
        )
        print(f"LR warmup: {cfg.warmup_epochs} epochs ({cfg.lr:.2e} target) + cosine decay to {float(cfg.lr)*0.01:.2e}")
    elif cfg.warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-6, end_factor=1.0, total_iters=int(cfg.warmup_epochs)
        )
        print(f"LR warmup: {cfg.warmup_epochs} epochs ({cfg.lr:.2e} target)")
    elif cfg.cosine_decay:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs, eta_min=float(cfg.lr) * 0.01
        )
        print(f"LR cosine decay: {cfg.lr:.2e} → {float(cfg.lr)*0.01:.2e} over {cfg.epochs} epochs")

    mech_names = ds.obs_names.tolist() if ds.obs_names is not None else None
    if cfg.obs_idx is not None and mech_names is not None:
        mech_names = [mech_names[i] for i in cfg.obs_idx]

    print(f"Data: N={N} | train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")
    print(f"Dims: P_obs={P_obs} | scaffold={cfg.scaffold} | U={U}")
    if mech_names is not None:
        print("Species:", ", ".join(str(x) for x in mech_names))

    best_val = float("inf")
    best_state = None

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_species_losses: list[np.ndarray] = []

    def _save_ckpt(path: Path, epoch: int, tag: str):
        torch.save(
            {
                "epoch": int(epoch),
                "tag": str(tag),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "opt_state": opt.state_dict(),
                "best_val": float(best_val),
                "cfg": cfg.__dict__,
            },
            path,
        )

    if cfg.obs_idx is not None:
        obs_idx = torch.tensor(cfg.obs_idx, device=device, dtype=torch.long)
        print(f"Supervising only species indices: {cfg.obs_idx}")
    else:
        obs_idx = torch.arange(P_obs, device=device, dtype=torch.long)

    dt_tensor = torch.from_numpy(ds.dt).to(device)
    grouped_model = cfg.model_class == "ode_transformer_grouped"

    species_weights: Optional[torch.Tensor] = None
    if bool(cfg.normalize_by_species_mean):
        w_np = compute_species_mean_weights(
            ds.y_seq, lengths=ds.lengths if ds.variable_length else None
        )
        # Index by obs_idx so shape aligns with pred/y_seq after subsetting.
        w_full = torch.from_numpy(w_np).to(device)
        species_weights = w_full[obs_idx]
        print(f"Species mean weights (obs): {species_weights.detach().cpu().tolist()}")

    r_terminal_enabled = float(cfg.lambda_r_terminal) > 0.0
    r_idx_full: int | None = None
    if r_terminal_enabled:
        state_names = getattr(scaffold, "state_names", None)
        if state_names is None or "R" not in list(state_names):
            raise ValueError(
                f"lambda_r_terminal>0 but scaffold {cfg.scaffold} has no 'R' state."
            )
        r_idx_full = list(state_names).index("R")
        print(f"R terminal reg: lambda={cfg.lambda_r_terminal}, frac={cfg.r_terminal_frac}, r_idx_full={r_idx_full}")

    for ep in range(1, cfg.epochs + 1):
        ep_t0 = time.time()
        teacher_forcing = bool(cfg.teacher_forcing) and (ep < int(cfg.tf_drop_epoch))

        # ---- train
        model.train()
        tr_total = 0.0
        tr_batches = 0

        for y0, u_seq, y_seq, batch_lengths in train_loader:
            K_batch = u_seq.shape[1]
            dt_seq = dt_tensor[:K_batch][None, :].expand(y0.shape[0], -1)

            y0 = y0.to(device)
            y_seq = y_seq.to(device)
            u_seq = u_seq.to(device)
            dt_seq = dt_seq.to(device)
            if batch_lengths is not None:
                batch_lengths = batch_lengths.to(device)

            opt.zero_grad(set_to_none=True)
            model_kwargs = {
                "teacher_forcing": teacher_forcing,
                "tf_every": int(cfg.tf_every),
            }
            if grouped_model and batch_lengths is not None:
                model_kwargs["lengths"] = batch_lengths
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.autocast_bf16 and device.type == "cuda")):
                pred, theta, _ = model(
                    y0,
                    u_seq,
                    dt_seq,
                    obs_idx,
                    y_seq,
                    **model_kwargs,
                )
            pred_full = pred
            pred = pred[:, :, obs_idx]
            y_seq = y_seq[:, :, obs_idx]

            loss = loss_fn(
                pred, y_seq, batch_lengths,
                loss_type=cfg.loss_type,
                species_weights=species_weights,
            )

            if r_terminal_enabled:
                r_loss = r_terminal_loss(
                    pred_full, r_idx_full, batch_lengths, float(cfg.r_terminal_frac)
                )
                loss = loss + float(cfg.lambda_r_terminal) * r_loss

            if cfg.l1_regularization:
                reg_loss = torch.mean(torch.abs(theta[:,1:,:] - theta[:,:-1,:]))
                loss += cfg.lambda_reg * reg_loss

            if cfg.l2_regularization:
                reg_loss = torch.mean((theta[:,1:,:] - theta[:,:-1,:]).pow(2))
                loss += cfg.lambda_reg * reg_loss

            loss.backward()

            if cfg.grad_clip and float(cfg.grad_clip) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))

            opt.step()

            tr_total += float(loss.item())
            tr_batches += 1

        tr_loss = tr_total / max(1, tr_batches)
        train_losses.append(tr_loss)

        if scheduler is not None:
            scheduler.step()

        # ---- val
        va_loss = None
        sp_last = None

        if val_loader is not None:
            model.eval()
            va_total = 0.0
            va_batches = 0
            sp_total = None
            r_start_sum = 0.0
            r_mid_sum = 0.0
            r_end_sum = 0.0
            r_min_sum = 0.0
            r_tail_sum = 0.0
            r_count = 0

            with torch.no_grad():
                for y0, u_seq, y_seq, batch_lengths in val_loader:
                    K_batch = u_seq.shape[1]
                    dt_seq = torch.from_numpy(ds.dt[:K_batch])
                    dt_seq = dt_seq[None, :].expand(y0.shape[0], -1)

                    y0 = y0.to(device)
                    y_seq = y_seq.to(device)
                    u_seq = u_seq.to(device)
                    dt_seq = dt_seq.to(device)
                    if batch_lengths is not None:
                        batch_lengths = batch_lengths.to(device)

                    model_kwargs = {"y_seq": None, "teacher_forcing": False}
                    if grouped_model and batch_lengths is not None:
                        model_kwargs["lengths"] = batch_lengths
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.autocast_bf16 and device.type == "cuda")):
                        pred, _, _ = model(y0, u_seq, dt_seq, obs_idx, **model_kwargs)

                    if r_terminal_enabled and r_idx_full is not None:
                        Bv, Kv = pred.shape[0], pred.shape[1]
                        r_traj = pred[:, :, r_idx_full].float()  # (B, K)
                        if batch_lengths is None:
                            lens_v = torch.full((Bv,), Kv, device=pred.device, dtype=torch.long)
                        else:
                            lens_v = batch_lengths.long()
                        idx_end = (lens_v - 1).clamp(min=0)
                        idx_mid = (lens_v // 2).clamp(min=0)
                        arange_b = torch.arange(Bv, device=pred.device)
                        r_start_sum += float(r_traj[:, 0].sum().item())
                        r_mid_sum   += float(r_traj[arange_b, idx_mid].sum().item())
                        r_end_sum   += float(r_traj[arange_b, idx_end].sum().item())
                        valid_mask = (torch.arange(Kv, device=pred.device)[None, :] < lens_v[:, None])
                        r_masked = torch.where(valid_mask, r_traj, torch.full_like(r_traj, float("inf")))
                        r_min_sum += float(r_masked.min(dim=1).values.sum().item())
                        tail_mask = _build_tail_mask(
                            lens_v, Kv, pred.device, float(cfg.r_terminal_frac), Bv
                        ).to(r_traj.dtype)
                        tail_denom = tail_mask.sum(dim=1).clamp(min=1.0)
                        r_tail_sum += float(((r_traj * tail_mask).sum(dim=1) / tail_denom).sum().item())
                        r_count += Bv

                    pred = pred[:, :, obs_idx]
                    y_seq = y_seq[:, :, obs_idx]

                    loss = loss_fn(
                        pred, y_seq, batch_lengths,
                        loss_type=cfg.loss_type,
                        species_weights=species_weights,
                    )
                    va_total += float(loss.item())

                    sp = loss_fn_per_species(
                        pred, y_seq, batch_lengths,
                        loss_type=cfg.loss_type,
                        species_weights=species_weights,
                    ).detach().cpu()
                    sp_total = sp if sp_total is None else sp_total + sp
                    va_batches += 1

            va_loss = va_total / max(1, va_batches)
            val_losses.append(va_loss)

            r_stats = None
            if r_count > 0:
                r_stats = {
                    "start": r_start_sum / r_count,
                    "mid":   r_mid_sum   / r_count,
                    "end":   r_end_sum   / r_count,
                    "min":   r_min_sum   / r_count,
                    "tail":  r_tail_sum  / r_count,
                }

            if sp_total is not None:
                sp_last = (sp_total / max(1, va_batches)).numpy()
                val_species_losses.append(sp_last)

            if va_loss < best_val:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        ep_time = time.time() - ep_t0

        if va_loss is None:
            print(f"ep {ep:4d} | train {tr_loss:.6f} | tf={int(teacher_forcing)} | {ep_time:.2f}s")
        else:
            sp_str = ""
            if sp_last is not None:
                if mech_names is None:
                    sp_str = "  [" + "  ".join(f"{v:.4f}" for v in sp_last) + "]"
                else:
                    sp_str = "  [" + "  ".join(f"{n}:{v:.4f}" for n, v in zip(mech_names, sp_last)) + "]"
            r_str = ""
            if r_stats is not None:
                r_str = (f"  R[t0={r_stats['start']:.3f} mid={r_stats['mid']:.3f} "
                         f"end={r_stats['end']:.3f} min={r_stats['min']:.3f} "
                         f"tail={r_stats['tail']:.3f}]")
            print(
                f"ep {ep:4d} | train {tr_loss:.6f} | val {va_loss:.6f} | best {best_val:.6f} | tf={int(teacher_forcing)}{sp_str}{r_str} | {ep_time:.2f}s"
            )

        if math.isnan(tr_loss) or math.isnan(va_loss):
            print(f"NaN detected at epoch {ep} — stopping early.")
            break

        if wandb_run is not None:
            payload = {
                "epoch": int(ep),
                "train/loss": float(tr_loss),
                "train/teacher_forcing": int(teacher_forcing),
                "system/epoch_time_sec": float(ep_time),
                "system/learning_rate": float(opt.param_groups[0]["lr"]),
            }
            if va_loss is not None:
                payload["val/loss"] = float(va_loss)
                payload["val/best_loss"] = float(best_val)
            if sp_last is not None:
                names = mech_names if mech_names is not None else [f"species_{i}" for i in range(len(sp_last))]
                for name, value in zip(names, sp_last):
                    payload[f"val_species/{name}"] = float(value)
            if r_stats is not None:
                for k, v in r_stats.items():
                    payload[f"val_R/{k}"] = float(v)
            wandb_run.log(payload, step=int(ep))

        # always keep "last" checkpoint
        _save_ckpt(exp_dir / cfg.save_last_name, ep, tag="last")

        # periodic checkpoints for epoch-evolution overlays
        if int(cfg.ckpt_every) > 0 and (ep % int(cfg.ckpt_every) == 0):
            _save_ckpt(ckpt_dir / f"ckpt_ep{ep:04d}.pt", ep, tag="periodic")

        # write curves every epoch (so final plotting can use full history)
        curves_path = logs_dir / cfg.save_curves_name
        np.savez(
            curves_path,
            train_losses=np.array(train_losses, dtype=np.float32),
            val_losses=np.array(val_losses, dtype=np.float32) if len(val_losses) > 0 else None,
            val_species_losses=np.array(val_species_losses, dtype=np.float32) if len(val_species_losses) > 0 else None,
        )

    # restore best weights (if we had validation)
    if best_state is not None:
        model.load_state_dict(best_state)

    # final test evaluation (on held-out test set, using best weights)
    test_loss: float | None = None
    test_species_loss: np.ndarray | None = None
    if len(test_idx) > 0:
        test_loader = DataLoader(
            torch.utils.data.Subset(ds, test_idx.tolist()),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=bool(cfg.pin_memory),
        )
        model.eval()
        te_total = 0.0
        te_batches = 0
        sp_total = None
        with torch.no_grad():
            for y0, u_seq, y_seq, batch_lengths in test_loader:
                K_batch = u_seq.shape[1]
                dt_seq = dt_tensor[:K_batch][None, :].expand(y0.shape[0], -1)
                y0 = y0.to(device)
                y_seq = y_seq.to(device)
                u_seq = u_seq.to(device)
                dt_seq = dt_seq.to(device)
                if batch_lengths is not None:
                    batch_lengths = batch_lengths.to(device)
                model_kwargs = {"y_seq": None, "teacher_forcing": False}
                if grouped_model and batch_lengths is not None:
                    model_kwargs["lengths"] = batch_lengths
                pred, _, _ = model(y0, u_seq, dt_seq, obs_idx, **model_kwargs)
                pred = pred[:, :, obs_idx]
                y_seq = y_seq[:, :, obs_idx]
                loss = loss_fn(
                    pred, y_seq, batch_lengths,
                    loss_type=cfg.loss_type,
                    species_weights=species_weights,
                )
                te_total += float(loss.item())
                sp = loss_fn_per_species(
                    pred, y_seq, batch_lengths,
                    loss_type=cfg.loss_type,
                    species_weights=species_weights,
                ).detach().cpu()
                sp_total = sp if sp_total is None else sp_total + sp
                te_batches += 1
        test_loss = te_total / max(1, te_batches)
        if sp_total is not None:
            test_species_loss = (sp_total / max(1, te_batches)).numpy()
            sp_str = "  [" + "  ".join(
                f"{n}:{v:.4f}" for n, v in zip(
                    mech_names if mech_names else [f"s{i}" for i in range(len(test_species_loss))],
                    test_species_loss,
                )
            ) + "]"
        else:
            sp_str = ""
        print(f"\nTest loss (best model): {test_loss:.6f}{sp_str}")

    # write final loss_curves.npz including test results
    np.savez(
        logs_dir / cfg.save_curves_name,
        train_losses=np.array(train_losses, dtype=np.float32),
        val_losses=np.array(val_losses, dtype=np.float32) if len(val_losses) > 0 else None,
        val_species_losses=np.array(val_species_losses, dtype=np.float32) if len(val_species_losses) > 0 else None,
        test_loss=np.float32(test_loss) if test_loss is not None else None,
        test_species_losses=test_species_loss.astype(np.float32) if test_species_loss is not None else None,
    )

    # save best model (plot expects exp_dir/model.pt)
    save_path = exp_dir / cfg.save_model_name
    torch.save(
        {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
         "best_val": float(best_val),
         "test_loss": float(test_loss) if test_loss is not None else None},
        save_path,
    )
    print(f"Saved best model to {save_path}")

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.2f}s ({elapsed/60:.2f}m)")
    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    # plots ONLY at the end (including epoch evolution overlays from checkpoints)
    if not no_plot:
        try:
            from plot_diagnostics import plot_experiment, plot_epoch_prediction_overlays

            plot_experiment(exp_dir, n_samples=int(plot_samples), sample_idx=int(plot_sample_idx))

            # epochs=None => automatically uses available checkpoints and picks up to max_overlays evenly spaced
            plot_epoch_prediction_overlays(
                exp_dir,
                sample_idx=int(plot_sample_idx),
                epochs=None,
                max_overlays=8,
            )
        except ImportError:
            print("[plot] plot_diagnostics.py not found; skipping plots.")
        except Exception as e:
            print(f"[plot] failed: {e}")

    if wandb_run is not None:
        plots_dir = exp_dir / "plots"
        log_wandb_images(wandb, wandb_run, plots_dir)
        log_wandb_artifact(wandb, wandb_run, exp_dir=exp_dir, run_id=run_id)
        wandb_run.summary["run_dir"] = str(exp_dir.resolve())
        wandb_run.summary["study"] = cfg.study
        wandb_run.summary["scaffold"] = cfg.scaffold
        wandb_run.summary["device"] = str(device)
        wandb_run.summary["elapsed_seconds"] = float(elapsed)
        if train_losses:
            wandb_run.summary["final_train_loss"] = float(train_losses[-1])
        if val_losses:
            wandb_run.summary["final_val_loss"] = float(val_losses[-1])
        if best_state is not None:
            wandb_run.summary["best_val_loss"] = float(best_val)
        wandb_run.finish()

    # Endpoint R^2 analysis (optional)
    if cfg.endpoint_r2:
        try:
            from metrics import endpoint_r2
            from plot_diagnostics import device_auto as _dev_auto

            r2_device = _dev_auto()
            print("\nRunning endpoint R^2 analysis...")
            result = endpoint_r2.collect_endpoints(
                exp_dir, r2_device, split="test", protein_sp="pm", mrna_sp="mm"
            )
            r2_protein = endpoint_r2.r2(result["true_protein_final"], result["pred_protein_final"])
            r2_mrna = endpoint_r2.r2(result["true_mrna_max"], result["pred_mrna_max"])
            print(f"  R²(protein final) = {r2_protein:.4f}")
            print(f"  R²(mRNA max)      = {r2_mrna:.4f}")

            out_path = exp_dir / "endpoint_r2.png"
            endpoint_r2.plot_endpoints([result], protein_sp="pm", mrna_sp="mm",
                                       split="test", out_path=out_path)

            if wandb_run is not None:
                wandb_run.summary["endpoint_r2/protein_final"] = float(r2_protein)
                wandb_run.summary["endpoint_r2/mrna_max"] = float(r2_mrna)
        except Exception as e:
            print(f"[endpoint_r2] failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-samples", type=int, default=5)
    parser.add_argument("--plot-sample-idx", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override any config field, e.g. --set lr=1e-3 --set scaffold=mof_synthesis_8")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    for kv in args.set:
        key, val = kv.split("=", 1)
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config field: {key!r}. Valid fields: {list(vars(cfg).keys())}")
        parsed = yaml.safe_load(val)
        # yaml.safe_load("None") → str "None", not Python None — fix explicitly
        if isinstance(parsed, str) and parsed == "None":
            parsed = None
        orig_type = type(getattr(cfg, key))
        if parsed is not None and orig_type is not type(None) and not isinstance(parsed, orig_type):
            parsed = orig_type(parsed)
        setattr(cfg, key, parsed)

    train(cfg, no_plot=bool(args.no_plot), plot_samples=int(args.plot_samples), plot_sample_idx=int(args.plot_sample_idx))
