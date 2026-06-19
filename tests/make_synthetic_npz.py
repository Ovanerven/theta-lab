"""Generate a tiny synthetic TXTL dataset npz for smoke-testing.

The real CFPS datasets are NOT shipped in this repo. This builds a small
npz with the SAME schema (7 observed channels ['R','O','m','mm','p','pm','DNA'],
13 control channels, per-sample dt grid, length/source metadata) so that
`train.py` can run end-to-end without any real data. The trajectories are
non-mechanistic toy curves — only the shapes/keys matter for a smoke test.
"""
from __future__ import annotations

import sys
import numpy as np

OBS_NAMES = ['R', 'O', 'm', 'mm', 'p', 'pm', 'DNA']
CONTROL_NAMES = ['3PGA', 'AA', 'DNA c', 'FB', 'K-Glut', 'Lysate 2%PEG',
                 'Maltose', 'Mg-Glut', 'NTPs', 'PEG8000', 'T7RNAP', 'water', 'u_open']
NAMES_FULL = ['R', 'O', 'm', 'mm', 'p', 'pm', 'DNA', '3PGA', 'AA', 'DNA c', 'FB',
              'K-Glut', 'Lysate 2%PEG', 'Maltose', 'Mg-Glut', 'NTPs', 'PEG8000',
              'T7RNAP', 'water']
CONTROL_INDICES = np.array([7, 8, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.int64)
OBS_INDICES = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)


def make_synthetic_npz(out_path: str, n: int = 300, k: int = 40, seed: int = 0) -> str:
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    P, U = 7, 13
    t = np.arange(k)

    y0 = np.zeros((n, P), np.float32)
    y0[:, 0] = 1.0  # R
    y0[:, 1] = 1.0  # O

    mm = (rng.uniform(50, 400, size=(n, 1)) * (1 - np.exp(-t / 12))[None, :]).astype(np.float32)
    pm = (rng.uniform(200, 3000, size=(n, 1)) * (1 - np.exp(-t / 20))[None, :]).astype(np.float32)
    y_seq = np.zeros((n, k, P), np.float32)
    y_seq[:, :, 3] = mm
    y_seq[:, :, 5] = pm
    y_seq[:, :, 0] = np.clip(1 - t / (2 * k), 0, 1)[None, :]
    y_seq[:, :, 1] = np.clip(1 - t / (3 * k), 0, 1)[None, :]

    u_seq = np.zeros((n, k, U), np.float32)
    u_seq[:, 0, 2] = rng.uniform(1, 5, size=n)   # DNA c bolus at t0
    u_seq[:, 0, 8] = rng.uniform(1, 3, size=n)   # NTPs

    dt_per_sample = np.full((n, k), 60.0, np.float32)
    t_obs = np.cumsum(np.full(k, 60.0)).astype(np.float32)
    lengths = np.full(n, k, np.int64)
    z_expr = np.ones(n, np.int64)
    half = n // 2
    source_label = np.array((['old'] * half) + (['new'] * (n - half)), dtype=object)

    np.savez(out_path,
             y0=y0, u_seq=u_seq, y_seq=y_seq, dt_per_sample=dt_per_sample,
             t_obs=t_obs, lengths=lengths, z_expr=z_expr, source_label=source_label,
             control_indices=CONTROL_INDICES, obs_indices=OBS_INDICES,
             control_names=np.array(CONTROL_NAMES), obs_names=np.array(OBS_NAMES),
             names_full=np.array(NAMES_FULL),
             n_states_full=np.int64(7), n_params_full=np.int64(0))
    return out_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_smoke.npz"
    print("wrote", make_synthetic_npz(out))
