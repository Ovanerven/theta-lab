#!/bin/bash
# Audit training run completion across all (or selected) experiments.
# Prints one line per INCOMPLETE run (missing model.pt).
#
# Usage:
#   bash scripts/check_run_status.sh               # all experiments
#   bash scripts/check_run_status.sh txtl_*        # glob pattern
#   bash scripts/check_run_status.sh txtl_gru_training_sweep txtl_bleach_lowfloor_sweep
#
# Exit code is the number of incomplete runs found.

set -euo pipefail
cd "$(dirname "$0")/.."   # always run from repo root

if [[ $# -eq 0 ]]; then
    studies=(experiments/*/)
else
    # Expand any globs the caller passed
    studies=()
    for arg in "$@"; do
        for d in experiments/${arg}/; do
            [[ -d "$d" ]] && studies+=("$d")
        done
    done
fi

n_incomplete=0
n_complete=0
n_studies=0

for study_dir in "${studies[@]}"; do
    study=$(basename "$study_dir")
    [[ "$study" == "old" || "$study" == "adhoc" ]] && continue

    runs=($(ls "$study_dir" 2>/dev/null))
    [[ ${#runs[@]} -eq 0 ]] && continue

    study_incomplete=0
    study_complete=0

    for run in "${runs[@]}"; do
        run_dir="${study_dir}${run}"
        [[ ! -d "$run_dir" ]] && continue

        if [[ -f "${run_dir}/model.pt" ]]; then
            (( study_complete++ )) || true
        else
            # Determine why it failed: has model_last.pt (trained but crashed at end),
            # has config.yaml (started), or empty (never started).
            if [[ -f "${run_dir}/model_last.pt" ]]; then
                reason="has model_last.pt but no model.pt (crashed after checkpoint)"
            elif [[ -f "${run_dir}/config.yaml" ]]; then
                reason="started (config exists) but no checkpoint — likely OOM/timeout"
            else
                reason="empty dir — job never started or was preempted immediately"
            fi
            echo "INCOMPLETE  $study / $run  [$reason]"
            (( study_incomplete++ )) || true
            (( n_incomplete++ )) || true
        fi
    done
    (( n_complete += study_complete )) || true
    (( n_studies++ )) || true
done

echo ""
echo "=== Summary ==="
echo "Studies checked : $n_studies"
echo "Complete runs   : $n_complete"
echo "Incomplete runs : $n_incomplete"

exit $n_incomplete
