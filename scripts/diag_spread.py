"""Mean-regression / dynamic-range diagnostic: does the model span the true output
range, or compress toward the mean? For each run dir (argv), compute on the TEST split:
  - std(pred)/std(true)  for pm-final  (1.0 = full spread; <1 = compressed)
  - OLS slope of pred~true (1.0 = ideal; <1 = regress-to-mean)
  overall and per source (old/new).  Compare native baseline vs coarse runs.
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/Users/olivervanerven/Documents/Thesis/theta-lab"); LLO = ROOT / "last-layer-ode"
sys.path.insert(0, str(LLO))
from plot_diagnostics import (rebuild_model_from_experiment, _test_subset, _maybe_lift,
                              _filter_model_kwargs, load_yaml)
from train import collate, collate_varlen
from metrics.endpoint_r2 import _apply_channel_min_gate as _gate
from scaffolds import SCAFFOLDS

def spread(exp):
    exp = Path(exp)
    model, ds, *_ , lift_info = rebuild_model_from_experiment(exp, torch.device("cpu"))
    model.eval(); lift_info = lift_info or {}
    cfg = load_yaml(exp / "config.yaml"); scaf = SCAFFOLDS[str(cfg["scaffold"])]
    # partial-obs scaffolds (M4/M3) put pm at obs_state_idx[1]; full-obs (M5) keeps
    # dataset layout where pm = cfg.obs_idx[1] (=5). pred/true share the post-lift layout.
    osi = getattr(scaf, "obs_state_idx", None)
    PM = int(osi[1]) if osi is not None else int(cfg["obs_idx"][1])
    raw = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    pds = _test_subset(ds, exp)
    idxs = list(pds.indices) if isinstance(pds, torch.utils.data.Subset) else list(range(len(pds)))
    src = np.array([int(raw.source_idx[i]) if raw.source_idx is not None else -1 for i in idxs])
    cf = collate_varlen if getattr(raw, "variable_length", False) else collate
    b = next(iter(torch.utils.data.DataLoader(pds, batch_size=len(idxs), shuffle=False, collate_fn=cf)))
    y0, u, y, Ls = b[0], b[1], b[2], b[3]
    dt = b[5] if len(b) >= 6 else torch.from_numpy(raw.dt[:u.shape[1]])[None].expand(y0.shape[0], -1)
    if bool(cfg.get("subtract_channel_min", False)):
        cols = cfg.get("subtract_channel_min_cols"); cols = [int(c) for c in cols] if cols else None
        y0, y = _gate(y0, y, cols, Ls)
    y0, y = _maybe_lift(y0, y, lift_info)
    oi = torch.tensor(lift_info["scaffold_obs_idx"], dtype=torch.long) if lift_info else torch.arange(y0.shape[-1])
    mk = {"y_seq": None, "teacher_forcing": False, "u_transform": str(cfg.get("u_transform", "none")),
          "y_transform": str(cfg.get("y_transform", "none"))}
    with torch.no_grad():
        pred, _, _ = model(y0, u, dt, oi, **_filter_model_kwargs(model, mk))
    pred = pred.cpu().numpy(); yt = y.cpu().numpy(); Ls = Ls.cpu().numpy()
    pt = np.array([yt[j, Ls[j]-1, PM] for j in range(len(idxs))])
    pp = np.array([pred[j, Ls[j]-1, PM] for j in range(len(idxs))])
    def stats(m):
        t, p = pt[m], pp[m]
        if len(t) < 3 or t.std() < 1e-6: return (np.nan,)*5
        slope = np.polyfit(t, p, 1)[0]
        r2 = 1.0 - np.sum((t-p)**2)/np.sum((t-t.mean())**2)
        big = t > 50.0   # leverage-free: relative error only where pm is meaningfully nonzero
        rel = np.abs(p[big] - t[big]) / t[big] if big.sum() else np.array([np.nan])
        return (r2, p.std()/t.std(), slope, float(np.median(rel)), float(np.mean(rel < 0.25)))
    print(f"\n{exp.name}")
    for lbl, m in [("ALL", np.ones(len(idxs), bool)), ("old", src==0), ("new", src==1)]:
        r2, sr, sl, mr, w = stats(m)
        print(f"   {lbl:4s}  R²={r2:.3f}  std(pred)/std(true)={sr:.3f}  slope={sl:.3f}  | median|relerr|={mr:.3f}  within25%={w:.2f}")

for e in sys.argv[1:]:
    try: spread(e)
    except Exception as ex: print(f"{e}: ERR {ex}")
