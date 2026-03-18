#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

CONFIG_DIR="configs/last-layer-ode-simple-scenario-all-scaffolds-2"
STUDY="last-layer-ode-simple-scenario-all-scaffolds-2"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIGS=(
  "reduced2.yaml"
  "reduced3.yaml"
  "reduced4_AEIM.yaml"
  "reduced5.yaml"
  "reduced6_ADGJLM.yaml"
  "reduced7.yaml"
  "reduced8_ADGHJKLM.yaml"
  "reduced9.yaml"
  "reduced10_with_M.yaml"
  "reduced11_with_M.yaml"
  "reduced12_with_M.yaml"
  "full13.yaml"
)

echo "Running ${#CONFIGS[@]} scaffold trainings from ${CONFIG_DIR}"

for cfg in "${CONFIGS[@]}"; do
  cfg_path="${CONFIG_DIR}/${cfg}"
  if [[ ! -f "$cfg_path" ]]; then
    echo "[ERROR] Missing config: $cfg_path" >&2
    exit 1
  fi

  echo
  echo "============================================================"
  echo "Training config: $cfg_path"
  echo "============================================================"
  "$PYTHON_BIN" last-layer-ode/train.py --config "$cfg_path"
done

EXP_BASE="experiments/${STUDY}"
if [[ ! -d "$EXP_BASE" ]]; then
  echo "[ERROR] Experiment base directory not found: $EXP_BASE" >&2
  exit 1
fi

LATEST_DATE_DIR="$(ls -1d "$EXP_BASE"/* 2>/dev/null | sort | tail -n 1)"
if [[ -z "${LATEST_DATE_DIR}" || ! -d "${LATEST_DATE_DIR}" ]]; then
  echo "[ERROR] Could not determine latest date folder in $EXP_BASE" >&2
  exit 1
fi

echo
echo "============================================================"
echo "Computing metrics in: ${LATEST_DATE_DIR}"
echo "============================================================"

PYTHONPATH=last-layer-ode "$PYTHON_BIN" -u last-layer-ode/metrics/summarize_nrmse.py \
"${LATEST_DATE_DIR}" \
--species A M \
--out "${LATEST_DATE_DIR}/nrmse_detailed.csv"

"$PYTHON_BIN" -u last-layer-ode/metrics/plot_nrmse.py \
"${LATEST_DATE_DIR}/nrmse_summary.csv" \
--stat mean \
--error-bar sem \
--format pdf \
--out "${LATEST_DATE_DIR}/nrmse_vs_P.pdf"

echo
echo "All scaffold trainings completed."
echo "Saved metrics to: ${LATEST_DATE_DIR}"