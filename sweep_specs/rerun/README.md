# Gap-filling rerun specs

Every spec here covers **only the runs missing** from the consolidated tables
(see `results/sweep_consolidated/all_results_long.csv` and `RESULTS_MANIFEST.md`).
Each one lands its outputs in the matching `experiments/<study>/` folder
alongside the existing runs, so the next `aggregate_from_cache.py` picks them up
automatically.

> **Env prerequisite:** the `.venv` torch install is broken (`torch/lib/`
> shared libraries are missing). Reinstall torch before running anything:
> `.venv/bin/pip install --force-reinstall torch==2.10.0`. Mamba runs also need
> the `mamba_env` conda env active.

## Architecture sweep — the gaps showing as `--` in Tables 7 & 8

| Spec | Runs | What it fills |
|---|---|---|
| `single_enzyme_lumped_arch_n3_missing.yaml`     | 4 | GRU, LSTM, Mamba, Transformer @ n=3 |
| `single_enzyme_lumped_arch_n10_missing.yaml`    | 4 | GRU, LSTM, Mamba, Transformer @ n=10 |
| `single_enzyme_lumped_arch_n1000_missing.yaml`  | 4 | GRU, LSTM, Mamba, Transformer @ n=1000 |
| `mof_synthesis_6_arch_n3_missing.yaml`          | 8 | GRU, LSTM, Mamba, Transformer × {full, first-last} @ n=3 |
| `mof_synthesis_6_arch_n10_missing.yaml`         | 5 | LSTM/fl, Transformer std+fl, Mamba std+fl @ n=10 |

Plus two specs that already exist in `sweep_specs/final/architecture_sweep/` with
the right runs uncommented — just launch them:

| Existing spec | Runs | What it fills |
|---|---|---|
| `glycolysis_reduced8_arch.yaml`         | 2 | sLSTM std+fl @ n=100 |
| `glycolysis_reduced8_arch_n1000.yaml`   | 2 | sLSTM std+fl @ n=1000 |

## New ablation — GRU-static-θ (Model B1) across every scaffold/n

`gru_static_theta_all_scaffolds.yaml` — **88 runs** (11 scaffolds × 4 n × {full,
first-last}). Uses the already-implemented `ode_rnn_sparse_theta_v2` with
`n_theta_anchors=1`, which makes the GRU encoder fire its θ-head **once at
step 0** and reuse that θ for the whole trajectory. Sits between A2 (IC θ via
MLP from `y0`) and A6 (CMVF, time-varying GRU θ).

Each run sets its own `study:` so outputs land in the matching
`experiments/<scaffold>_data_ablation/` folder alongside A1..A9. The new row
labelled **GRU-static (θ)** picks up automatically the next time you run
`aggregate_from_cache.py + make_sweep_tables.py + build_table_pdfs.sh`.

Optional — run only a subset (e.g. n=3 sanity check on one scaffold) by
duplicating the spec and trimming the `runs:` list.

## Data ablation — small targeted refills

| Spec | Runs | What it fills |
|---|---|---|
| `glycolysis_full22_data_n1000_missing.yaml` | 2 | A9_neural_ode_correction std+FL @ n=1000 (the only truly-failed cells) |
| (existing) `sweep_specs/final/retry/single_enzyme_lumped_retry.yaml` | 6 | re-score of `first_last_n1000` runs that have `model.pt` but no `nrmse_cache.csv` |

## Run-all (local, no SLURM)

```bash
# 1) repair torch (one-off)
.venv/bin/pip install --force-reinstall torch==2.10.0

# 2) activate the env that owns torch (and mamba_env when launching mamba runs)
source $(conda info --base)/etc/profile.d/conda.sh && conda activate thesis_env

# 3) launch each rerun spec (max_parallel = how many you can fit on your GPU)
for f in sweep_specs/rerun/*.yaml; do
  python launch_sweep.py "$f" --local --max-parallel 1 --no-compare
done

# 4) refresh the consolidated CSVs + tables + preview PDFs
.venv/bin/python scripts/aggregate_from_cache.py
.venv/bin/python scripts/make_sweep_tables.py
bash scripts/build_table_pdfs.sh
```

For Mamba-only runs the launch must happen inside `mamba_env`:
`conda activate mamba_env && python launch_sweep.py <spec> --local …`.
