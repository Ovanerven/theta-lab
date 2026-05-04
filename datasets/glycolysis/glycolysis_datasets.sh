#!/usr/bin/env bash
set -euo pipefail

# Glycolysis datasets for each scaffold/model.
#
# Goal: obs_indices match the scaffold state ordering (P matches scaffold.P)
# and control_indices match the input maps defined in `last-layer-ode/sim/glycolysis.py`.
#
# Models available via `last-layer-ode/create_dataset.py --model ...`:
#   - glycolysis_oracle22   (P=22)
#   - glycolysis_reduced12  (P=12)
#   - glycolysis_reduced8   (P=8)
#   - glycolysis_reduced4   (P=4)

T_SPAN=20
N_STEPS=400
SEED=42

# N values to generate
NS=(1000 100 10 3)

# -----------------------------------------------------------------------------
# ORACLE22 scaffold: observe all 22 states; controls per GLYCOLYSIS_ORACLE22_INPUT_MAP
# -----------------------------------------------------------------------------
ORACLE_CONTROL_INDICES="0,12,13,14,15,16,17,18,19,20,21"
ORACLE_OBS_INDICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21"

for N in "${NS[@]}"; do
  python last-layer-ode/create_dataset.py --model glycolysis_oracle22 \
    --t-span ${T_SPAN} --n-steps ${N_STEPS} --n-samples ${N} \
    --control-indices ${ORACLE_CONTROL_INDICES} \
    --obs-indices ${ORACLE_OBS_INDICES} \
    --seed ${SEED} \
    --output-file datasets/glycolysis/glycolysis_oracle22_n${N}.npz
done

# -----------------------------------------------------------------------------
# REDUCED12 scaffold: observe all 12 states; controls per GLYCOLYSIS_REDUCED12_INPUT_MAP
# -----------------------------------------------------------------------------
R12_CONTROL_INDICES="0,8,9,10,11"   # Glc, ATP, ADP, NAD, NADH
R12_OBS_INDICES="0,1,2,3,4,5,6,7,8,9,10,11"

for N in "${NS[@]}"; do
  python last-layer-ode/create_dataset.py --model glycolysis_reduced12 \
    --t-span ${T_SPAN} --n-steps ${N_STEPS} --n-samples ${N} \
    --control-indices ${R12_CONTROL_INDICES} \
    --obs-indices ${R12_OBS_INDICES} \
    --seed ${SEED} \
    --output-file datasets/glycolysis/glycolysis_reduced12_n${N}.npz
done

# -----------------------------------------------------------------------------
# REDUCED8 scaffold: observe all 8 states; controls per GLYCOLYSIS_REDUCED8_INPUT_MAP
# -----------------------------------------------------------------------------
R8_CONTROL_INDICES="0,5,6,7"        # Glc, ATP, NAD, NADH
R8_OBS_INDICES="0,1,2,3,4,5,6,7"

for N in "${NS[@]}"; do
  python last-layer-ode/create_dataset.py --model glycolysis_reduced8 \
    --t-span ${T_SPAN} --n-steps ${N_STEPS} --n-samples ${N} \
    --control-indices ${R8_CONTROL_INDICES} \
    --obs-indices ${R8_OBS_INDICES} \
    --seed ${SEED} \
    --output-file datasets/glycolysis/glycolysis_reduced8_n${N}.npz
done

# -----------------------------------------------------------------------------
# REDUCED4 scaffold: observe all 4 states; controls per GLYCOLYSIS_REDUCED4_INPUT_MAP
# -----------------------------------------------------------------------------
R4_CONTROL_INDICES="0,2,3"          # Glc, ATP, NADH
R4_OBS_INDICES="0,1,2,3"

for N in "${NS[@]}"; do
  python last-layer-ode/create_dataset.py --model glycolysis_reduced4 \
    --t-span ${T_SPAN} --n-steps ${N_STEPS} --n-samples ${N} \
    --control-indices ${R4_CONTROL_INDICES} \
    --obs-indices ${R4_OBS_INDICES} \
    --seed ${SEED} \
    --output-file datasets/glycolysis/glycolysis_reduced4_n${N}.npz
done
