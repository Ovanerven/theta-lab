"""Decompose B-run endpoint R² by source (OLD vs NEW test samples).

Self-contained loader that handles the fact that historical experiments were
trained on 12-col or 13-col u_seq while the current NPZ has 14 cols (is_new +
u_open were added later). We detect U_ckpt from the checkpoint's u_to_y_jump
shape and slice the dataset's u_seq accordingly so the loaded model is fed
exactly the columns it was trained on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "last-layer-ode"))
from train import ODEDataset, TrainConfig
import scaffolds as sf
from models.ode_rnn import OdeRNN
from jumps import make_u_to_y_jump
import yaml


def load_yaml(p): return yaml.safe_load(open(p))


def r2(y_true, y_pred):
    if len(y_true) < 2:
        return float("nan")
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / (ss_tot + 1e-12) if ss_tot > 0 else float("nan")


def decompose(exp_dir: Path, device: torch.device):
    cfg = load_yaml(exp_dir / "config.yaml")
    ds_path = Path(cfg["dataset_path"])
    if not ds_path.is_absolute():
        for c in [Path("."), Path("/home/overven/theta-lab")]:
            if (c / ds_path).exists():
                ds_path = c / ds_path; break
    ds = ODEDataset(str(ds_path))
    # source_label not exposed on ODEDataset; load directly from the npz file.
    _raw = np.load(ds_path, allow_pickle=True)
    src = np.array([str(s) for s in _raw["source_label"]]) if "source_label" in _raw.files else None
    if src is None:
        print("  dataset has no source_label field; cannot decompose")
        return

    # Detect U the checkpoint expects from u_to_y_jump.
    ckpt = torch.load(exp_dir / "model.pt", map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    U_ckpt = int(sd["u_to_y_jump"].shape[0])
    U_cur = ds.u_seq.shape[-1]
    print(f"  U_ckpt={U_ckpt}, U_current_dataset={U_cur}  →  slicing u_seq to [:, :, :{U_ckpt}]")

    # Build matching scaffold/model. We assume scaffold P matches dataset P_obs
    # (no partial-obs lift on these B-arm experiments — they use Model 5).
    scaf = sf.SCAFFOLDS[cfg["scaffold"]]
    obs_idx_list = list(cfg.get("obs_idx", [3, 5]))

    # Sub-select dataset control_indices to U_ckpt (preserves the "first U_ckpt
    # columns are the original reagent set" invariant).
    ctl_idx_sliced = np.asarray(ds.control_indices, dtype=np.int64)[:U_ckpt]
    obs_idx_arr = np.asarray(ds.obs_indices, dtype=np.int64)
    u_to_y_jump = make_u_to_y_jump(ctl_idx_sliced, obs_idx_arr).to(device)

    # Pull every kwarg the OdeRNN constructor cares about from cfg so the model
    # matches what was trained (lift_skip, gru_u/y_cols, num_layers, transforms).
    kwargs = dict(
        rhs=scaf, U=U_ckpt, P=scaf.P,
        hidden=int(cfg.get("hidden", 600)),
        u_to_y_jump=u_to_y_jump.cpu(),
        num_layers=int(cfg.get("num_layers", 1)),
        dropout=float(cfg.get("dropout", 0.0)),
        gru_u_cols=cfg.get("gru_u_cols"),
        gru_y_cols=cfg.get("gru_y_cols"),
        lift_dim=int(cfg.get("lift_dim", 32)),
        lift_skip=bool(cfg.get("lift_skip", False)),
        gru_variant=str(cfg.get("gru_variant", "rnn")),
        encoder_use_time=bool(cfg.get("encoder_use_time", False)),
        encoder_use_log_dt=bool(cfg.get("encoder_use_log_dt", False)),
        y0_theta_init=bool(cfg.get("y0_theta_init", False)),
        theta_head_transform=str(cfg.get("theta_head_transform", "log_gamma")),
        theta_head_tau=float(cfg.get("theta_head_tau", 1.0)),
        use_basal=bool(cfg.get("use_basal", False)),
        detach_y_prev=bool(cfg.get("detach_y_prev", False)),
        tf_at_k_zero=bool(cfg.get("tf_at_k_zero", False)),
    )
    # Drop kwargs OdeRNN doesn't accept (older repo versions vary).
    import inspect as _ins
    sig = _ins.signature(OdeRNN.__init__)
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters or k == "rhs"}
    model = OdeRNN(**kwargs)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:    print(f"  missing keys:    {len(missing)} (first 3: {missing[:3]})")
    if unexpected: print(f"  unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    model.to(device); model.eval()

    # Iterate test split
    split = np.load(exp_dir / "split.npz")
    test_idx = split["test_idx"].astype(int).tolist()

    rows = []  # (gi, source, true_pm, pred_pm, true_mm_max, pred_mm_max)
    z_arr = ds.z_expr if ds.z_expr is not None else None
    with torch.no_grad():
        for gi in test_idx:
            if z_arr is not None and int(z_arr[gi]) == 0:
                continue  # skip synth
            item = ds[gi]
            y0, u_seq_full, y_seq, dt_i = item[0], item[1], item[2], item[3]
            # Slice u_seq to the columns the checkpoint expects.
            u_seq = u_seq_full[..., :U_ckpt]
            y0_b = y0.unsqueeze(0).to(device)
            u_b = u_seq.unsqueeze(0).to(device)
            dt_b = dt_i.unsqueeze(0).to(device)
            obs_idx_t = torch.tensor(obs_idx_list, device=device, dtype=torch.long)
            pred, _, _ = model(
                y0_b, u_b, dt_b, obs_idx_t,
                y_seq=None, teacher_forcing=False,
                u_transform=str(cfg.get("u_transform", "none")),
                y_transform=str(cfg.get("y_transform", "none")),
            )
            y_np = y_seq.cpu().numpy(); p_np = pred[0].cpu().numpy()
            p_idx = obs_idx_list[1]  # pm position in dataset/scaffold (same here)
            m_idx = obs_idx_list[0]
            rows.append((gi, src[gi],
                         float(y_np[-1, p_idx]), float(p_np[-1, p_idx]),
                         float(y_np[:, m_idx].max()), float(p_np[:, m_idx].max())))

    if not rows:
        print("  no test rows scored"); return
    sources = np.array([r[1] for r in rows])
    arr = np.array([(r[2], r[3], r[4], r[5]) for r in rows])
    print(f"  total test rows scored: {len(rows)}")
    for label in ["all", "old", "new"]:
        mask = np.ones(len(rows), bool) if label == "all" else sources == label
        n = int(mask.sum())
        if n < 2:
            print(f"  {label:>4}: n={n}  (too few)")
            continue
        tpf, ppf, tmm, pmm = arr[mask, 0], arr[mask, 1], arr[mask, 2], arr[mask, 3]
        print(f"  {label:>4}: n={n:3d}  R²(pm_final)={r2(tpf, ppf):+.3f}  R²(mm_max)={r2(tmm, pmm):+.3f}  "
              f"|  median true_pm={np.median(tpf):7.1f}  pred_pm={np.median(ppf):7.1f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = [
        Path("experiments/txtl_data_axis_native_gru_noy0_drecipe/20260524_000846_B_native_real"),
        Path("experiments/txtl_data_axis_native_gru/20260524_110742_B_native_real"),
        Path("experiments/txtl_data_axis_native_gru/20260524_110735_B_native_real_logdt"),
    ]
    for t in targets:
        if not t.exists():
            print(f"missing: {t}"); continue
        print(f"\n=== {t.name} ===")
        try:
            decompose(t, device)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR: {e}")
