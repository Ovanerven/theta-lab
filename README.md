# Bolus-to-parameter maps

All runnable scripts in the last-layer-ode directory. Run everything from the **repo root**.

---

## Training

| Script | Purpose |
|---|---|
| `train.py` | Train a model. Takes `--config` and `--set key=val` overrides. |

---

## Evaluation  ← start here after training

| Script | Purpose |
|---|---|
| `metrics/compare_runs.py` | Compare all runs in a folder by NRMSE. Computes on first run, caches to `nrmse_cache.csv`. |
| `metrics/plot_nrmse.py` | Plot NRMSE vs scaffold size (P). Same cache as `compare_runs.py`. |
| `metrics/plot_sample_fit_grid.py` | Plot true vs predicted trajectories for one sample across all runs in a folder. |
| `plot_diagnostics.py` | Full diagnostic plots for a single run (loss curves, species heatmap, theta, etc.). Also used as a library by other scripts. |

**Typical evaluation workflow:**
```bash
# Compare runs — first call computes NRMSE and saves nrmse_cache.csv
python last-layer-ode/metrics/compare_runs.py experiments/scaffold_size_effect

# Plot NRMSE vs P — reuses the cache
python last-layer-ode/metrics/plot_nrmse.py experiments/scaffold_size_effect --out results/scaffold_size_effect.pdf

# Visual fit check for one sample
python last-layer-ode/metrics/plot_sample_fit_grid.py experiments/scaffold_size_effect

# Force recompute (e.g. after retraining)
python last-layer-ode/metrics/compare_runs.py experiments/scaffold_size_effect --recompute
```

---

## Analysis

| Script | Purpose |
|---|---|
| `analysis/per_step_theta_fit.py` | Fit theta parameters per time step via gradient descent on a single trajectory. |
| `analysis/plot_per_step_theta_fit.py` | Plot results from `per_step_theta_fit.py` (NRMSE vs P, truth vs rollout grid). |
| `analysis/observability_experiment.py` | Observability experiment (which species are needed for identifiability). |

---

## Baselines

| Script | Purpose |
|---|---|
| `baselines/honest_rollout.py` | Honest autoregressive rollout baseline (no teacher forcing). |
| `baselines/manual_theta_fit.py` | Manual / analytical theta fitting baseline. |
| `baselines/neural_ode.py` | Pure neural ODE baseline (no mechanistic structure). |
| `baselines/ode_rnn_analytical.py` | ODE-RNN with analytical ODE layer. |
| `baselines/ode_rnn_og.py` | Original ODE-RNN (Rubanova et al. 2019) baseline. |

---

## Data

| Script | Purpose |
|---|---|
| `create_dataset.py` | Simulate and save a dataset (npz). |
| `preview_dataset.py` | Quick visual check of a dataset file. |
| `sim/MOF_model.py` | MOF synthesis ODE model definition. |
| `sim/syndata_simulator_ODE.py` | General ODE simulator for synthetic data. |
| `sim/Single_enzyme.py` | Single-enzyme kinetics model. |
| `sim/benchmark_models.py` | Benchmark model implementations. |

---

## Libraries (not runnable directly)

| File | Purpose |
|---|---|
| `models/` | Model implementations: `ode_rnn`, `ode_transformer`, `ode_mlp`, `ode_rnn_2020`, `neural_ode`. |
| `scaffolds.py` | Scaffold (mechanistic ODE structure) registry. |
| `jumps.py` | Control-to-state jump mappings. |
