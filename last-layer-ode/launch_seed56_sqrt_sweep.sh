#!/usr/bin/env bash
# Real grid sweep over the axes that disambiguate why training collapses to mean
# while Bob's checkpoint reaches R²≈0.7. Each axis is meaningful on its own:
#
#   U_TRANSFORMS    sqrt | minmax_sqrt
#                   sqrt matches Bob's `torch.sqrt(u_k)` on raw values;
#                   minmax_sqrt is your sp12 default. If sqrt wins consistently,
#                   the input-scale hypothesis is confirmed.
#
#   SPLIT_SEEDS     56 | 12
#                   56 is Bob's published seed; 12 is your sp12. Tells us
#                   whether the data split itself is doing the work.
#
#   TRAIN_SEEDS     1 2 3
#                   Three training seeds per (u_transform, split_seed) cell to
#                   separate signal from variance-collapse jitter.
#
# Total: 2 × 2 × 3 = 12 runs. Sequential by default; --parallel forks all.
#
# Run:  bash launch_seed56_sqrt_sweep.sh
#       bash launch_seed56_sqrt_sweep.sh --parallel
#       bash launch_seed56_sqrt_sweep.sh --u "sqrt" --split "56" --train "1 2 3"
set -euo pipefail
cd "$(dirname "$0")"

CFG=../configs/sweep_seed56_sqrt.yaml
U_TRANSFORMS="sqrt minmax_sqrt"
SPLIT_SEEDS="56 12"
TRAIN_SEEDS="1 2 3"
PARALLEL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --parallel) PARALLEL=1; shift ;;
    --u) U_TRANSFORMS="$2"; shift 2 ;;
    --split) SPLIT_SEEDS="$2"; shift 2 ;;
    --train) TRAIN_SEEDS="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

OUT_ROOT=../experiments/bob_seed56_sqrt_sweep
mkdir -p "${OUT_ROOT}"

run_one() {
  local utr=$1 split=$2 trseed=$3
  local exp_name="u-${utr}_split${split}_seed${trseed}"
  echo "[launch] ${exp_name}"
  python3 train_R.py --config "${CFG}" \
    --set u_transform="${utr}" \
    --set split_seed="${split}" \
    --set seed="${trseed}" \
    --set exp_name="${exp_name}" \
    --no-plot 2>&1 \
    | tee "${OUT_ROOT}/_log_${exp_name}.txt"
}

cells=()
for utr in $U_TRANSFORMS; do
  for split in $SPLIT_SEEDS; do
    for trs in $TRAIN_SEEDS; do
      cells+=("${utr}|${split}|${trs}")
    done
  done
done
echo "[plan] ${#cells[@]} runs:"
for c in "${cells[@]}"; do echo "  - $c"; done

if [ "$PARALLEL" = "1" ]; then
  pids=()
  for c in "${cells[@]}"; do
    IFS='|' read -r utr split trs <<< "$c"
    run_one "$utr" "$split" "$trs" & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
else
  for c in "${cells[@]}"; do
    IFS='|' read -r utr split trs <<< "$c"
    run_one "$utr" "$split" "$trs"
  done
fi

echo "[done] sweep complete; aggregate with:"
echo "  python3 -c \"import csv,glob; rows=[]; [rows.extend(list(csv.DictReader(open(f)))) for f in glob.glob('${OUT_ROOT}/*/r2_cache.csv')]; [print(r) for r in rows]\""
