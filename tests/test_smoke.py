"""End-to-end smoke test: train each shipped config for 1 epoch on a tiny
synthetic dataset and assert it completes and writes a model checkpoint.

No real data and no pytest required — run directly:

    python tests/test_smoke.py

Exits 0 on success, non-zero on failure.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CODE_DIR = REPO / "last-layer-ode"
CONFIGS = REPO / "configs"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_synthetic_npz import make_synthetic_npz


def _run_one(config_name: str, npz_path: str, out_root: str) -> Path:
    """Run train.py for 1 epoch on the tiny dataset; return the run dir."""
    cfg = CONFIGS / config_name
    assert cfg.exists(), f"missing config {cfg}"
    cmd = [
        sys.executable, "-u", "train.py", "--config", str(cfg), "--no-plot",
        "--set", f"dataset_path={npz_path}",
        "--set", "epochs=1", "--set", "warmup_epochs=0",
        "--set", "val_n=4", "--set", "test_n=4", "--set", "batch_size=8",
        "--set", "num_workers=0", "--set", "pin_memory=false",
        "--set", "jit_scripting=false", "--set", "endpoint_r2=false",
        "--set", f"out_root={out_root}", "--set", "study=smoke",
        "--set", f"exp_name={Path(config_name).stem}",
    ]
    print(f"\n=== {config_name} ===\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(CODE_DIR), capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-2000:])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        raise AssertionError(f"{config_name}: train.py exited {proc.returncode}")
    runs = list((Path(out_root) / "smoke").glob("*"))
    assert runs, f"{config_name}: no run dir created under {out_root}/smoke"
    model = runs[0] / "model.pt"
    assert model.exists(), f"{config_name}: model.pt not written ({model})"
    print(f"OK — {config_name}: model saved at {model}")
    return runs[0]


def test_smoke_M5_gru():
    with tempfile.TemporaryDirectory() as td:
        npz = make_synthetic_npz(str(Path(td) / "synth.npz"), n=24)
        _run_one("scaffold_ladder_gru_M5.yaml", npz, str(Path(td) / "exp"))


def test_smoke_M4_slstm():
    with tempfile.TemporaryDirectory() as td:
        npz = make_synthetic_npz(str(Path(td) / "synth.npz"), n=24)
        _run_one("encoder_zoo_slstm_M4.yaml", npz, str(Path(td) / "exp"))


if __name__ == "__main__":
    test_smoke_M5_gru()
    test_smoke_M4_slstm()
    print("\nAll smoke tests passed.")
