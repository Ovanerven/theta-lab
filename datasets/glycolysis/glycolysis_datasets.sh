#!/usr/bin/env bash
set -euo pipefail

# Glycolysis datasets — ALL generated from the 22-state oracle.
#
# Reduced scaffolds (4/8/12) are fit to oracle-generated data so the experiment
# is meaningful: scaffold has fewer states + simplified kinetics, data has full
# 22-state structure. Lumping is applied at save time inside create_dataset.py
# (see _oracle_lumping_for_reduced{4,8,12}); the simulator is always oracle22.
#
# Models routed to the oracle simulator:
#   glycolysis_oracle22                — 22 obs (full state)
#   glycolysis_oracle_to_reduced12     — 12 obs (HexP=G6P+F6P+FBP, TriP=GAP+DHAP, PG3=PG3+PG2)
#   glycolysis_oracle_to_reduced8      —  8 obs (SugarP = all phosphorylated intermediates)
#   glycolysis_oracle_to_reduced4      —  4 obs (Glc, Pyr, ATP, NADH; clean subset)

T_SPAN=20
N_STEPS=400
SEED=42

# N values to generate
NS=(1000 100 10 3)

# Oracle22 controls (same for all reduced datasets — trajectory is from oracle22):
#   Glc + ATP/ADP/NAD/NADH + 6 inhibitor pools
ORACLE_CONTROL_INDICES="0,12,13,14,15,16,17,18,19,20,21"

# -----------------------------------------------------------------------------
# ORACLE22 — full 22-state observed
# -----------------------------------------------------------------------------
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
# REDUCED12 from oracle — 12 lumped channels (HexP, TriP, PG3 lumped)
# obs_indices are the *primary* oracle22 index per lumped state (used only for
# control→state jump wiring; the actual y_seq is M @ x_full).
# Order matches reduced12 scaffold: [Glc, HexP, TriP, BPG13, PG3, PEP, Pyr, Lac,
#                                    ATP, ADP, NAD, NADH]
# -----------------------------------------------------------------------------
R12_OBS_INDICES="0,1,4,6,7,9,10,11,12,13,14,15"

for N in "${NS[@]}"; do
  python last-layer-ode/create_dataset.py --model glycolysis_oracle_to_reduced12 \
    --t-span ${T_SPAN} --n-steps ${N_STEPS} --n-samples ${N} \
    --control-indices ${ORACLE_CONTROL_INDICES} \
    --obs-indices ${R12_OBS_INDICES} \
    --seed ${SEED} \
    --output-file datasets/glycolysis/glycolysis_reduced12_n${N}.npz
done

# -----------------------------------------------------------------------------
# REDUCED8 from oracle — 8 lumped channels (SugarP = idx 1..8)
# Order matches reduced8 scaffold: [Glc, SugarP, PEP, Pyr, Lac, ATP, NAD, NADH]
# -----------------------------------------------------------------------------
R8_OBS_INDICES="0,1,9,10,11,12,14,15"

for N in "${NS[@]}"; do
  python last-layer-ode/create_dataset.py --model glycolysis_oracle_to_reduced8 \
    --t-span ${T_SPAN} --n-steps ${N_STEPS} --n-samples ${N} \
    --control-indices ${ORACLE_CONTROL_INDICES} \
    --obs-indices ${R8_OBS_INDICES} \
    --seed ${SEED} \
    --output-file datasets/glycolysis/glycolysis_reduced8_n${N}.npz
done

# -----------------------------------------------------------------------------
# REDUCED4 from oracle — clean 4-state subset [Glc, Pyr, ATP, NADH]
# No lumping needed: oracle22 + obs-indices does it directly.
# -----------------------------------------------------------------------------
R4_OBS_INDICES="0,10,12,15"

for N in "${NS[@]}"; do
  python last-layer-ode/create_dataset.py --model glycolysis_oracle22 \
    --t-span ${T_SPAN} --n-steps ${N_STEPS} --n-samples ${N} \
    --control-indices ${ORACLE_CONTROL_INDICES} \
    --obs-indices ${R4_OBS_INDICES} \
    --seed ${SEED} \
    --output-file datasets/glycolysis/glycolysis_reduced4_n${N}.npz
done
