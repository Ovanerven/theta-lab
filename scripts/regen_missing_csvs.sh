#!/bin/bash
# Resubmit compare jobs for all experiments that are missing a CSV.
# Run from the theta-lab root:
#   bash scripts/regen_missing_csvs.sh [--dry-run]
#
# Each study gets one sbatch job via compare.job (thesis_env, gpu_a100).
# endpoint_r2 flag is set per-study based on whether endpoint_r2=true appears
# in any run config under experiments/<STUDY>/.

set -euo pipefail
cd "$(dirname "$0")/.."   # always run from repo root

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# --- Determine if a study needs --endpoint-r2 ---
needs_endpoint_r2() {
    local study="$1"
    local first_cfg
    first_cfg=$(find "experiments/$study" -name "config.yaml" -maxdepth 2 | head -1)
    if [[ -n "$first_cfg" ]] && grep -q "endpoint_r2: true" "$first_cfg" 2>/dev/null; then
        echo "1"
    else
        echo "0"
    fi
}

# --- Check if a study has a complete CSV already ---
has_csv() {
    local study="$1"
    [[ -f "results/${study}.csv" || -f "results/${study}_mean.csv" || -d "results/${study}" ]]
}

# --- Count completed runs (have model.pt) vs total ---
run_status() {
    local study="$1"
    local total complete
    total=$(ls "experiments/$study" 2>/dev/null | wc -l)
    complete=$(find "experiments/$study" -name "model.pt" -maxdepth 2 2>/dev/null | wc -l)
    echo "${complete}/${total}"
}

# --- Submit or print one compare job ---
submit_compare() {
    local study="$1"
    local ep_r2="$2"
    local extra_time="${3:-01:00:00}"

    local export_str="ALL,STUDY=${study}"
    [[ "$ep_r2" == "1" ]] && export_str="${export_str},ENDPOINT_R2=1"

    local cmd=(
        sbatch
        "--job-name=${study}_compare"
        "--time=${extra_time}"
        "--output=slurm_outputs/${study}/%A_%x.out"
        "--export=${export_str}"
        "slurm_jobs/compare.job"
    )

    mkdir -p "slurm_outputs/${study}"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] ${cmd[*]}"
    else
        jid=$("${cmd[@]}" | awk '{print $NF}')
        echo "  submitted job $jid"
    fi
}

echo "=== Regenerating missing CSVs ==="
echo "(run with --dry-run to preview without submitting)"
echo ""

# Studies with many runs (63) that previously timed out → give 3h
LONG_STUDIES="txtl_gru_training_sweep txtl_init_loss_tf_sweep txtl_init_loss_tf_sweep_rerun_sqrt_utrans"

for study_dir in experiments/*/; do
    study=$(basename "$study_dir")
    [[ "$study" == "old" || "$study" == "adhoc" ]] && continue

    if has_csv "$study"; then
        continue
    fi

    status=$(run_status "$study")
    complete="${status%%/*}"
    total="${status##*/}"

    if [[ "$total" -eq 0 ]]; then
        echo "SKIP $study  (no runs at all)"
        continue
    fi

    ep_r2=$(needs_endpoint_r2 "$study")
    extra_time="01:00:00"
    for ls in $LONG_STUDIES; do
        [[ "$study" == "$ls" ]] && extra_time="03:00:00"
    done

    echo "$study  [${complete}/${total} complete]  endpoint_r2=${ep_r2}  time=${extra_time}"
    submit_compare "$study" "$ep_r2" "$extra_time"
done
