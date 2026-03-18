#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

CONFIG_DIR="configs/neural-ode-simple-scenario-all-Ps"
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

DEFAULT_STUDY_DIR="experiments/neural-ode-simple-scenario-all-Ps"
STUDY_DIR="${STUDY_DIR:-$DEFAULT_STUDY_DIR}"
DATE_ARG="${1:-}"
DROP_DIVERGED_ARG="${2:-}"

echo "[0/3] Training neural ODE baseline configs from: $CONFIG_DIR"
for cfg in "${CONFIGS[@]}"; do
  cfg_path="$CONFIG_DIR/$cfg"
  if [[ ! -f "$cfg_path" ]]; then
    echo "[error] Missing config: $cfg_path"
    exit 1
  fi
  echo "  - training $cfg"
  "$PYTHON_BIN" last-layer-ode/train.py --config "$cfg_path"
done
echo "[info] Training complete."

if [[ ! -d "$STUDY_DIR" ]]; then
  echo "[error] Study directory not found: $STUDY_DIR"
  echo "Run neural ODE baseline training first, or pass a custom path:"
  echo "  STUDY_DIR=experiments/<your-neural-study-dir> ./plot_neural_ode_baseline.sh"
  exit 1
fi

if [[ -n "$DATE_ARG" ]]; then
  TARGET_DIR="$STUDY_DIR/$DATE_ARG"
else
  TARGET_DIR="$(ls -1d "$STUDY_DIR"/*(/N) | sort | tail -n 1)"
fi

if [[ -z "${TARGET_DIR:-}" || ! -d "$TARGET_DIR" ]]; then
  echo "[error] Could not resolve a valid date folder under $STUDY_DIR"
  echo "Tip: pass a date explicitly, e.g. ./plot_neural_ode_baseline.sh 2026-03-17"
  exit 1
fi

echo "[info] Using experiment folder: $TARGET_DIR"

DROP_ARGS=()
if [[ -n "$DROP_DIVERGED_ARG" ]]; then
  DROP_ARGS=(--drop-diverged "$DROP_DIVERGED_ARG")
  echo "[info] Dropping trajectories with NRMSE > $DROP_DIVERGED_ARG"
fi

echo "[1/3] Summarizing NRMSE (A, M)..."
PYTHONPATH=last-layer-ode python -u last-layer-ode/metrics/summarize_nrmse.py \
  "$TARGET_DIR" \
  --species A M \
  "${DROP_ARGS[@]}" \
  --out "$TARGET_DIR/nrmse_detailed.csv"

echo "[2/3] Plotting NRMSE vs P (mean ± sem)..."
python -u last-layer-ode/metrics/plot_nrmse.py \
  "$TARGET_DIR/nrmse_detailed.csv" \
  --stat mean \
  --error-bar sem \
  --format pdf \
  --out "$TARGET_DIR/nrmse_vs_P_mean_sem.pdf"

echo "[done] Wrote:"
echo "  - $TARGET_DIR/nrmse_detailed.csv"
echo "  - $TARGET_DIR/nrmse_vs_P_mean_sem.pdf"
echo ""
echo "Usage:"
echo "  ./plot_neural_ode_baseline.sh [YYYY-MM-DD] [DROP_DIVERGED]"
echo "  example: ./plot_neural_ode_baseline.sh 2026-03-17 100"
echo "  PYTHON_BIN=python3 ./plot_neural_ode_baseline.sh"
echo "  STUDY_DIR=experiments/neural-ode-simple-scenario-all-Ps ./plot_neural_ode_baseline.sh"
