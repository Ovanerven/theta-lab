#!/bin/bash
# Architecture ablation sweep for paper table.
#
# 7 trainable ablations × 4 scaffold sizes = 28 jobs.
# Oracle ablation (per-step GD theta) is handled separately via manual_theta_fit.job.
#
# Ablations:
#   A1  ode_fixed_theta    — single global learnable theta, no NN
#   A2  ode_sample_theta   — per-sample constant theta (MLP encodes y0), no time variation
#   A3  neural_ode_gru     — GRU predicts dy/dt directly (black-box, Euler)       [already done]
#   A4  neural_ode_mlp     — MLP predicts dy/dt directly (black-box, no history)  [already done]
#   A5  ode_rnn            — baseline (GRU → bounded theta → scaffold + RK4)
#   A6  ode_rnn            — unbounded theta (softplus instead of gamma)           [already done]
#   A7  ode_rnn            — baseline + L1 theta regularization
#
# NOTE: A3, A4, A6 do not use gamma bounds and were already run correctly.
#       A1, A2, A5, A7 use per-parameter log-gamma bounds — requires correct theta_lo/theta_hi
#       per scaffold (set in scaffolds.py). Re-run these after updating bounds.
#
# Usage (from repo root):
#   bash slurm_jobs/sweeps/architecture_ablation.sh                  # seed 42 (original)
#   bash slurm_jobs/sweeps/architecture_ablation.sh experiments 1    # seed 1 → MOF_architecture_ablation_2_seed1
#   bash slurm_jobs/sweeps/architecture_ablation.sh experiments 2    # seed 2 → MOF_architecture_ablation_2_seed2

set -e

OUT_ROOT="${1:-experiments}"
SEED="${2:-42}"

if [ "${SEED}" = "42" ]; then
    STUDY="MOF_architecture_ablation_2"
else
    STUDY="MOF_architecture_ablation_2_seed${SEED}"
fi

mkdir -p "slurm_outputs/${STUDY}"

JOB_IDS=""

submit() {
    sbatch "$@" | awk '{print $4}'
}

append_id() {
    local id="$1"
    JOB_IDS="${JOB_IDS:+${JOB_IDS}:}${id}"
}

for SCAFFOLD in 4 6 8 12; do
    DATASET="datasets/mof_synthesis_${SCAFFOLD}.npz"
    SNAME="mof_synthesis_${SCAFFOLD}"

    # A1 — fixed global theta (no NN)
    JID=$(submit \
        --job-name="A1_fixed_${SCAFFOLD}" \
        --time="02:00:00" \
        --output="slurm_outputs/${STUDY}/%A_%x.out" \
        slurm_jobs/gpu.job \
        --config configs/archs/gru.yaml \
        --set study=${STUDY} \
        --set seed=${SEED} \
        --set out_root=${OUT_ROOT} \
        --set scaffold=${SNAME} \
        --set dataset_path=${DATASET} \
        --set exp_name=A1_fixed_theta_${SCAFFOLD} \
        --set model_class=ode_fixed_theta)
    append_id $JID
    echo "Submitted A1_fixed_theta_${SCAFFOLD} (job $JID)"

    # A2 — per-sample constant theta (MLP encodes y0, fixed across timesteps)
    JID=$(submit \
        --job-name="A2_sample_${SCAFFOLD}" \
        --time="02:00:00" \
        --output="slurm_outputs/${STUDY}/%A_%x.out" \
        slurm_jobs/gpu.job \
        --config configs/archs/gru.yaml \
        --set study=${STUDY} \
        --set seed=${SEED} \
        --set out_root=${OUT_ROOT} \
        --set scaffold=${SNAME} \
        --set dataset_path=${DATASET} \
        --set exp_name=A2_sample_theta_${SCAFFOLD} \
        --set model_class=ode_sample_theta)
    append_id $JID
    echo "Submitted A2_sample_theta_${SCAFFOLD} (job $JID)"

    # A3 — black-box GRU (dy/dt, Euler, no scaffold) [already done — commented out]
    JID=$(submit \
        --job-name="A3_gru_${SCAFFOLD}" \
        --time="02:00:00" \
        --output="slurm_outputs/${STUDY}/%A_%x.out" \
        slurm_jobs/gpu.job \
        --config configs/archs/gru.yaml \
        --set study=${STUDY} \
        --set seed=${SEED} \
        --set out_root=${OUT_ROOT} \
        --set scaffold=${SNAME} \
        --set dataset_path=${DATASET} \
        --set exp_name=A3_neural_ode_gru_${SCAFFOLD} \
        --set model_class=neural_ode_gru)
    append_id $JID
    echo "Submitted A3_neural_ode_gru_${SCAFFOLD} (job $JID)"

    # A4 — black-box MLP (dy/dt, RK4, no scaffold, no history) [already done — commented out]
    JID=$(submit \
        --job-name="A4_mlp_${SCAFFOLD}" \
        --time="02:00:00" \
        --output="slurm_outputs/${STUDY}/%A_%x.out" \
        slurm_jobs/gpu.job \
        --config configs/archs/gru.yaml \
        --set study=${STUDY} \
        --set seed=${SEED} \
        --set out_root=${OUT_ROOT} \
        --set scaffold=${SNAME} \
        --set dataset_path=${DATASET} \
        --set exp_name=A4_neural_ode_mlp_${SCAFFOLD} \
        --set model_class=neural_ode_mlp)
    append_id $JID
    echo "Submitted A4_neural_ode_mlp_${SCAFFOLD} (job $JID)"

    # A5 — baseline: GRU + bounded theta + scaffold
    JID=$(submit \
        --job-name="A5_baseline_${SCAFFOLD}" \
        --time="02:00:00" \
        --output="slurm_outputs/${STUDY}/%A_%x.out" \
        slurm_jobs/gpu.job \
        --config configs/archs/gru.yaml \
        --set study=${STUDY} \
        --set seed=${SEED} \
        --set out_root=${OUT_ROOT} \
        --set scaffold=${SNAME} \
        --set dataset_path=${DATASET} \
        --set exp_name=A5_ode_rnn_baseline_${SCAFFOLD} \
        --set model_class=ode_rnn)
    append_id $JID
    echo "Submitted A5_ode_rnn_baseline_${SCAFFOLD} (job $JID)"

    # A6 — baseline with unbounded theta (softplus) [already done — commented out]
    JID=$(submit \
        --job-name="A6_unbounded_${SCAFFOLD}" \
        --time="02:00:00" \
        --output="slurm_outputs/${STUDY}/%A_%x.out" \
        slurm_jobs/gpu.job \
        --config configs/archs/gru.yaml \
        --set study=${STUDY} \
        --set seed=${SEED} \
        --set out_root=${OUT_ROOT} \
        --set scaffold=${SNAME} \
        --set dataset_path=${DATASET} \
        --set exp_name=A6_ode_rnn_unbounded_${SCAFFOLD} \
        --set model_class=ode_rnn \
        --set theta_bounded=false)
    append_id $JID
    echo "Submitted A6_ode_rnn_unbounded_${SCAFFOLD} (job $JID)"

    # A7 — baseline with L1 theta regularization
    JID=$(submit \
        --job-name="A7_l1reg_${SCAFFOLD}" \
        --time="02:00:00" \
        --output="slurm_outputs/${STUDY}/%A_%x.out" \
        slurm_jobs/gpu.job \
        --config configs/archs/gru.yaml \
        --set study=${STUDY} \
        --set seed=${SEED} \
        --set out_root=${OUT_ROOT} \
        --set scaffold=${SNAME} \
        --set dataset_path=${DATASET} \
        --set exp_name=A7_ode_rnn_l1reg_${SCAFFOLD} \
        --set model_class=ode_rnn \
        --set l1_regularization=true)
    append_id $JID
    echo "Submitted A7_ode_rnn_l1reg_${SCAFFOLD} (job $JID)"

done

echo ""
echo "All 28 jobs submitted → ${OUT_ROOT}/${STUDY} (seed=${SEED})"
echo "Job IDs: ${JOB_IDS}"

# Submit compare job once all training jobs complete
CMP_JID=$(sbatch \
    --job-name="ablation_compare" \
    --dependency=afterok:${JOB_IDS} \
    --time="01:00:00" \
    --output="slurm_outputs/${STUDY}/%A_%x.out" \
    --export=ALL,STUDY=${STUDY} \
    slurm_jobs/compare.job | awk '{print $4}')

echo "Submitted compare job (job ${CMP_JID}) — runs after all training jobs complete"
