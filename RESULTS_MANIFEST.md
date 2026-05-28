# Results Manifest — Data Ablation & Architecture Sweep

_Audit date: 2026-05-28. Catalogue of every data-ablation and
architecture-sweep study, with expected-vs-actual run counts, the aggregated
CSV for each, the exact runs still missing, and the consolidated CSVs + LaTeX
appendix tables built from them._

## TL;DR — what was produced (2026-05-28)

- **`scripts/aggregate_from_cache.py`** — torch-free re-aggregation straight
  from each run's `nrmse_cache.csv` (works even though this machine's torch is
  broken). Rebuilds every per-study CSV and one consolidated long table, and
  validates each run's name against its dataset. Run: `.venv/bin/python scripts/aggregate_from_cache.py`.
- **`results/sweep_consolidated/csv/`** — 62 clean per-study `_{median,mean}.csv`.
- **`results/sweep_consolidated/all_results_long.csv`** — 853-row tidy table
  (family, study, scaffold, model, supervision, n, seed, nrmse_median,
  nrmse_mean, …). The single source of truth for the tables.
- **`scripts/make_sweep_tables.py`** — builds A4 LaTeX appendix tables in two
  layouts × two metrics under **`results/sweep_consolidated/tables/`**
  (`per_scaffold/` and `per_system/`, each with `median/` and `mean/`). Both
  compile clean on A4 (per_scaffold portrait; per_system landscape, chunked to
  ≤3 scaffolds/table + shrink-only `\resizebox`). Every folder has a `_preview.md`
  you can read without LaTeX. Model labels follow the thesis convention:
  data-ablation = CMVF / CMVF-L1 / CMVF-L2 / CMVF-unbounded / NODE-GRU / NODE-MLP
  / NODE-correction / Global ($\theta$) / Initial-condition ($\theta$);
  arch-sweep = GRU / LSTM / sLSTM / Transformer / Mamba.
- **`scripts/build_table_pdfs.sh`** — compiles each layout/metric into
  `results/sweep_consolidated/tables/preview_<layout>_<metric>.pdf` (run AFTER
  `make_sweep_tables.py`, which wipes the tables folder). All four currently
  build with 0 overfull boxes.

## Naming audit (the "missing `_nX`" worry) — resolved

- **No experiment *directory* in any of the 16 sweep studies is mis-named.**
  `aggregate_from_cache.py` cross-checks each run's name `_nX` against the
  `dataset_path` in its `config.yaml` (ground truth, per your tip) and reports
  **0 name/n mismatches**.
- The 73 runs that lack an `_nX` token are **all in IVTT/TXTL folders**
  (`ivtt_*`, `transformer_test`, `txtl_ladder_*`) — a different part of the
  project, not this sweep.
- The only artifact of past mis-naming was a **stale CSV row** `A6_baseline`
  (no `_n1000`) in the old `mof_synthesis_4_arch_sweep_n1000` CSV. The dir was
  long since renamed to `A6_baseline_n1000`; regenerating the CSVs fixes it.
  `A6_baseline` (= the GRU run pasted in as the arch-sweep GRU baseline) is now
  **relabeled "GRU"** in all arch-sweep tables.
- If a genuinely mis-named dir ever appears, the aggregator flags it
  (`NAME LACKS _n` / `N MISMATCH`) so you can rename the dir and re-run.

## The aggregation recipe (confirmed)

Implemented in `last-layer-ode/metrics/nrmse.py` + `compare_runs.py`:

1. Roll out the model on the **test split** of each run.
2. Per trajectory, per species → `NRMSE = RMSE / range(y_true)`.
   Non-finite preds (NaN / exploded) are replaced with a large **finite**
   sentinel so a diverged trajectory scores "huge but finite" instead of
   poisoning the aggregate.
3. **Median across trajectories**, per species → cached in each run's
   `nrmse_cache.csv` (also stores mean/std/q25/q75).
4. **Mean across species** → the `nrmse` column of the summary CSV.

`<study>_median.csv` and `<study>_mean.csv` differ only in whether step 3 uses
the median or mean across trajectories.

## Environment note (blocking)

The transferred `.venv` has a **broken torch install** — `torch/lib/` (the
shared libraries) is missing, so `import torch` fails on this machine. Nothing
that rebuilds a model (re-scoring or retraining) can run until this is fixed:

```bash
.venv/bin/pip install --force-reinstall torch==2.10.0
# then sanity check:
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

All "run locally" commands below assume a working torch env is active.

---

## Expected counts

| Study type | Per-scaffold target | Breakdown |
|---|---|---|
| Data ablation | **64** | 8 archs (A1,A2,A3,A4,A5,A6,A7,A9) × {std, first_last} × {n3,10,100,1000} |
| Arch sweep (MOF / glycolysis) | **40** | 5 archs (GRU, LSTM, sLSTM, Transformer, Mamba) × {std, fl} × {n3,10,100,1000} |
| Arch sweep (single-enzyme) | **20** | same 5 archs × {n3,10,100,1000}; **fl ≡ std** (only 2 species, so first_last = full obs) |

No mis-named/`n`-less test runs were found in any of these folders — every run
carries an `_n###` token, and the n=100 arch-sweep sets correctly live in the
un-suffixed base folders.

---

## 1. Data ablation — 9 of 11 complete ✅

CSVs live in `final_data_ablation/` (and a mirror in `results/final_data_ablation/`),
except `glycolysis_full22` which is in `results/`.

| Scaffold | CSV rows / 64 | Status |
|---|---|---|
| single_enzyme_4 | 64 | ✅ complete |
| single_enzyme_6 | 64 | ✅ complete |
| mof_synthesis_4 | 64 | ✅ complete (1 leftover un-cached duplicate dir, harmless) |
| mof_synthesis_6 | 64 | ✅ complete |
| mof_synthesis_8 | 64 | ✅ complete |
| mof_synthesis_12 | 64 | ✅ complete (8 leftover failed `A1/A2 first_last` dirs w/o model.pt — junk, CSV is fine) |
| glycolysis_reduced4 | 64 | ✅ complete |
| glycolysis_reduced8 | 64 | ✅ complete |
| glycolysis_reduced12 | 64 | ✅ complete |
| **single_enzyme_lumped** | **59** | ⚠️ **re-score only** — see 1a |
| **glycolysis_full22** | **60** | ❌ **4 failed runs need retraining** — see 1b |

### 1a. single_enzyme_lumped — re-score (no retraining)

These 6 runs have `model.pt` but no `nrmse_cache.csv`, so they were never
aggregated (all `first_last_n1000`):
`A3_neural_ode_gru`, `A4_neural_ode_mlp`, `A5_unbounded`, `A6_baseline`,
`A7_l1reg`, `A9_neural_ode_correction`.

Fix — just recompute caches + regenerate the CSV (rebuilds models, no training):

```bash
python last-layer-ode/metrics/compare_runs.py \
  experiments/single_enzyme_lumped_data_ablation \
  --csv final_data_ablation/single_enzyme_lumped_data_ablation.csv
# writes *_median.csv and *_mean.csv; copy to results/final_data_ablation/ to match the mirror
```

(If any model fails to load, the ready-made `sweep_specs/final/retry/single_enzyme_lumped_retry.yaml`
retrains exactly these.)

### 1b. glycolysis_full22 — retrain 4 failed cells

No `model.pt` for these `n1000` cells (training failed): `A1_fixed_theta/FL`,
`A2_sample_theta/FL`, `A9_neural_ode_correction/std`, `A9_neural_ode_correction/FL`.
All four are present in `sweep_specs/final/data_ablation_ran/glycolysis_oracle22.yaml`
(study `glycolysis_full22_data_ablation`, dataset
`datasets/glycolysis/glycolysis_oracle22_n1000.npz`, exists locally).

Make a 4-run rerun spec (copy those 4 lines into a new `runs:` list) and:

```bash
python launch_sweep.py <your_full22_rerun_spec>.yaml --local --max-parallel 2 --no-compare
python last-layer-ode/metrics/compare_runs.py \
  experiments/glycolysis_full22_data_ablation --csv results/glycolysis_full22_data_ablation.csv
```

---

## 2. Architecture sweep — 2 of 5 complete

Base folder (no suffix) = n100. CSVs in `results/` as
`<study>_arch_sweep[_n{3,10,1000}]_{median,mean}.csv`. Specs in
`sweep_specs/final/architecture_sweep/`. In the specs the GRU/LSTM/Mamba/Transformer
runs are **commented out** (they ran in an earlier pass) and only sLSTM is active;
fill gaps by **uncommenting the needed lines**. `obs_idx: [0,N]` = first_last.

| Scaffold | n3 | n10 | n100 | n1000 | Target | Status |
|---|---|---|---|---|---|---|
| mof_synthesis_4 | 10 | 10 | 10 | 10 | 40 | ✅ complete |
| glycolysis_reduced12 | 10 | 10 | 10 | 10 | 40 | ✅ complete |
| **mof_synthesis_6** | 2 | 5 | 10 | 10 | 40 | ❌ ~13 missing — see 2a |
| **glycolysis_reduced8** | 10 | 10 | (8) | 8 | 40 | ❌ 4 missing + CSV bug — see 2b |
| **single_enzyme_lumped** | 1 | 1 | 6 | 1 | 20 | ❌ ~12 missing — see 2c |

### 2a. mof_synthesis_6 — fill n3 and n10

- **n3** (`mof_synthesis_6_arch_n3.yaml`): only sLSTM (std+fl) present. Uncomment
  GRU, LSTM, Mamba, Transformer in **both** obs blocks → 8 runs.
- **n10** (`mof_synthesis_6_arch_n10.yaml`): missing LSTM/fl, Transformer (std+fl),
  Mamba (std+fl) → 5 runs.

```bash
python launch_sweep.py sweep_specs/final/architecture_sweep/mof_synthesis_6_arch_n3.yaml  --local --max-parallel 2 --no-compare
python launch_sweep.py sweep_specs/final/architecture_sweep/mof_synthesis_6_arch_n10.yaml --local --max-parallel 2 --no-compare
python last-layer-ode/metrics/compare_runs.py experiments/mof_synthesis_6_arch_sweep_n3   --csv results/mof_synthesis_6_arch_sweep_n3.csv
python last-layer-ode/metrics/compare_runs.py experiments/mof_synthesis_6_arch_sweep_n10  --csv results/mof_synthesis_6_arch_sweep_n10.csv
```

### 2b. glycolysis_reduced8 — run sLSTM + regenerate n100 CSV

sLSTM (std+fl) is missing at n100 and n1000 (4 runs); the sLSTM lines are
already uncommented in `glycolysis_reduced8_arch.yaml` (n100) and
`glycolysis_reduced8_arch_n1000.yaml`. **Also: the n100 summary CSV
(`glycolysis_reduced8_arch_sweep_median.csv`) was never generated** even though
the 8 non-sLSTM runs exist — regenerate it.

```bash
python launch_sweep.py sweep_specs/final/architecture_sweep/glycolysis_reduced8_arch.yaml       --local --max-parallel 2 --no-compare
python launch_sweep.py sweep_specs/final/architecture_sweep/glycolysis_reduced8_arch_n1000.yaml --local --max-parallel 2 --no-compare
python last-layer-ode/metrics/compare_runs.py experiments/glycolysis_reduced8_arch_sweep       --csv results/glycolysis_reduced8_arch_sweep.csv
python last-layer-ode/metrics/compare_runs.py experiments/glycolysis_reduced8_arch_sweep_n1000 --csv results/glycolysis_reduced8_arch_sweep_n1000.csv
```

### 2c. single_enzyme_lumped — fill GRU/LSTM/Transformer/Mamba at n3, n10, n1000

Only sLSTM ran across all n; n100 also has exploratory GRU/LSTM/Trans/Mamba.
Target is 20 (fl≡std). Missing = {GRU, LSTM, Transformer, Mamba} × {n3, n10, n1000}
= 12 runs. Uncomment those archs in:
`single_enzyme_lumped_arch_n3.yaml`, `_n10.yaml`, `_n1000.yaml`.

```bash
for n in n3 n10 n1000; do
  python launch_sweep.py sweep_specs/final/architecture_sweep/single_enzyme_lumped_arch_$n.yaml --local --max-parallel 2 --no-compare
  python last-layer-ode/metrics/compare_runs.py experiments/single_enzyme_lumped_arch_sweep_$n --csv results/single_enzyme_lumped_arch_sweep_$n.csv
done
```

---

## Caveats

- **Mamba** runs need the `mamba_env` conda env (specs tag them `env: mamba_env`).
  `--local` mode uses whatever env is active, so activate `mamba_env` for those.
- `results/final_single_enzyme_gru_*.csv` is a **separate legacy** single-enzyme
  study (2-state A/C, uses an `A8_l2reg` arch, no n-sweep) — not part of the
  data-ablation n-sweep above. Don't conflate.
- mof_12 / full22 have leftover failed/duplicate run dirs (no `model.pt`); they
  don't affect the CSVs and can be ignored or deleted.
