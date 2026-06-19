# Cleaned codebase — what changed vs `last-layer-ode/`

This folder is a pruned copy of `last-layer-ode/` for the final submission. The
original is untouched so you can diff the two. Everything kept here is used by
the canonical runs in `experiments_final/FINAL/`. The clean copy still imports
and still loads the original `config.yaml` files (unknown/removed keys are
ignored with a warning).

## Models (`models/`)
**Kept** (the model zoo actually used): `ode_rnn` (GRU CMVF), `ode_rnn_sparse_theta`
(K-anchor CMVF), encoder ablations `ode_transformer / ode_slstm / ode_mingru /
ode_lmu / ode_mamba (+ ode_mamba_ssm / ode_mambapy)`, `lstm_rnn`, the NODE
baselines `neural_ode_mlp / neural_ode_gru / neural_ode_correction`, and the
`ode_fixed_theta` static/global-θ baseline.

**Removed** (not used by any FINAL run):
- `ode_rnn_analytical.py`, `ode_rnn_txtl.py` — analytical-solution models (no longer used).
- `bob_gru_verbatim.py` — supervisor-reference model.
- `ode_rnn_basal_v2.py`, `ode_rnn_sparse_theta_v2.py`, `ode_sample_theta.py` — superseded experiments.
- `ode_mlstm.py` (already disabled), `ode_liquid.py` (never registered).
- In `ode_rnn.py`: deleted `StackedGRUCellBlock` and the `gru_variant="stacked_cell"`
  path (the old stacked GRU-cell-with-dropout-every-layer variant). Only `nn_gru` remains.
- In `ode_rnn_sparse_theta.py`: dropped the unused `OdesLSTMSparseTheta` and
  `OdeRNNBasalV2SparseTheta` wrappers; kept only `OdeRNNSparseTheta`.

## Scaffolds (`scaffolds.py`, 3541 → 496 lines)
Kept the 6 canonical TXTL scaffolds used in the final ladder (these also cover
every `use_synthetic_data` ablation run):
`txtl_model3_two_state` (M3), `txtl_model4_three_state` (M4),
`txtl_resource_and_maturation_dna` (M5), `txtl_model7_bg_fixed` (M7),
`txtl_model8_bg_fixed` (M8), `txtl_model9_event_dark` (M9).

Removed: all deprecated M7/M8/M9 variants (boundary_gated, kfix, fixed,
reagent_resource, peaked_fixed, oxygen_dark, dark_stable, o2a/o2b, dark_m4, …),
the other-domain scaffolds (MOF, single-enzyme, methane, glycolysis, kovacs,
westbrook, global_one_step), and the analytical IVTT scaffolds.

## Data generation (`sim/`, `create_dataset.py`)
The TXTL datasets are built by the repo-root `scripts/` (e.g. `build_txtl_combined_npz.py`).
`create_dataset.py` only generated the non-TXTL domains, so it and its sim deps
were removed: `MOF_model.py`, `Single_enzyme.py`, `explicit_methane_models.py`,
`benchmark_models.py`, `glycolysis.py`, `aranco_full_model.py`,
`syndata_simulator_ODE.py`. Kept `sim/txtl_synthetic_no_go.py`.

## Training (`train.py`)
- Config loader is now lenient: keys not in `TrainConfig` are ignored with a
  warning (so old configs with removed flags still load).
- Removed the deprecated `loss_fn_ivtt_mse` path and its `use_ivtt_mse_loss` flag.
- Removed dead dispatch for deleted models (`bob_gru_verbatim` guard,
  `ode_*_sparse_theta_v2`, the extra sparse-θ wrapper names) and the
  `legacy_forget_bias_bug` flag.

## Comments / naming
- Removed supervisor-/“Bob”-attribution and stale dev-log comments.
- The model init option value `"supervisor"` (for `gru_init` / `head_init`) was
  renamed to the neutral `"orthogonal"`; `"supervisor"` is kept as a
  backward-compatible alias so existing configs run **bit-identically**.
- Deleted deprecated standalone scripts: `train_basal_v2.py`,
  `analysis/per_step_theta_fit_analytical.py`.

## Standalone reference scripts
`standalone/` contains two self-contained training+eval scripts (architecture +
training loop + R² eval in one file each) for the two best models — see
`standalone/README.md`.
