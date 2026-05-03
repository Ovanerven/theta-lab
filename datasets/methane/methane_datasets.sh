# Methane datasets — generate from 3 simulators (GRI30, Kazakov, Smooke)
# Each with its own recommended observed species (7 per model).
#
# Controls (same for all): feed_CH4, feed_O2, feed_N2, feed_H2O
# Timescale: t_span=5, n_steps=200 (stoichiometric CH4/O2/N2 plateaus by t~5)
#
# Observed species per model (from explicit_methane_models.py):
#   GRI30:   CH4, O2, CO, CO2, H2O, OH, NO
#   Kazakov: CH4, O2, CO, CO2, H2O, OH, NO
#   Smooke:  CH4, O2, CO, CO2, H2O, OH, CH2O

# --- Dataset 1: GRI30 Full (53-state simulator) -> 7 observed species ---
# GRI30 species indices:
#   CH4=13, O2=3, CO=14, CO2=15, H2O=5, OH=4, NO=35
# Controls: feed_CH4=13, feed_O2=3, feed_N2=47, feed_H2O=5
python last-layer-ode/create_dataset.py --model gri30_full \
    --t-span 5 --n-steps 200 --n-samples 1000 \
    --control-indices 13,3,47,5 \
    --obs-indices 13,3,14,15,5,4,35 \
    --output-file datasets/gri30_obs7.npz

# --- Dataset 2: Kazakov Middle (28-state simulator) -> 7 observed species ---
# Kazakov species indices (local to Kazakov's 28 species):
#   CH4=11, O2=3, CO=12, CO2=13, H2O=5, OH=4, NO=24
# Controls: feed_CH4, feed_O2, feed_N2, feed_H2O (mapped to Kazakov inputs)
python last-layer-ode/create_dataset.py --model kazakov_middle \
    --t-span 5 --n-steps 200 --n-samples 1000 \
    --control-indices 11,3,22,5 \
    --obs-indices 11,3,12,13,5,4,24 \
    --output-file datasets/kazakov_obs7.npz

# --- Dataset 3: Smooke Reduced (16-state simulator) -> 7 observed species ---
# Smooke species indices (local to Smooke's 16 species):
#   CH4=0, O2=2, CO=9, CO2=14, H2O=8, OH=5, CH2O=11
# Controls: feed_CH4, feed_O2, feed_N2, feed_H2O (mapped to Smooke inputs)
python last-layer-ode/create_dataset.py --model smooke_reduced_model \
    --t-span 5 --n-steps 200 --n-samples 1000 \
    --control-indices 0,2,15,8 \
    --obs-indices 0,2,9,14,8,5,11 \
    --output-file datasets/smooke_obs7.npz

# --- Sliced subsets of the 5-obs GRI30 dataset (no re-simulation) ---
# Source 5-obs ordering: [CH4, O2, CO, CO2, H2O]  (GRI30 idx 13,3,14,15,5)
#
# westbrook_dryer_2step states: [CH4, O2, CO, CO2]   -> keep cols 0,1,2,3
# global_one_step       states: [CH4, O2, CO2]       -> keep cols 0,1,3
python datasets/methane/slice_obs.py \
    --source datasets/gri30_obs5_real_rates.npz \
    --keep-cols 0,1,2,3 \
    --output datasets/gri30_obs4_westbrookdryer.npz

python datasets/methane/slice_obs.py \
    --source datasets/gri30_obs5_real_rates.npz \
    --keep-cols 0,1,3 \
    --output datasets/gri30_obs3_global1step.npz

# --- Quick smoke test (N=5) ---
# python last-layer-ode/create_dataset.py --model gri30_full \
#     --t-span 5 --n-steps 200 --n-samples 5 \
#     --control-indices 13,3,47,5 \
#     --obs-indices 13,3,14,15,5,4,35 \
#     --output-file datasets/gri30_obs7_test.npz

# ARAMCO!!

# This one was too easy
python last-layer-ode/create_dataset.py                                                                                          
    --model aramco_30
    --t-span 5 \ 
    --n-steps 200— \
    --n-samples 1000 \ 
    --control-indices 15,5,1,7 \
    --obs-indices 15,5,7,12,13,3,4,6,8,11,10,16,28,1 \
    --output-file datasets/aramco_kovacs14_hard.npz

# This one has temperature changing per sample
python last-layer-ode/create_dataset.py \
    --model aramco_30 \
    --t-span 5 \
    --n-steps 200 \
    --n-samples 1000 \
    --control-indices 15,5,1,7 \
    --obs-indices 15,5,7,12,13,3,4,6,8,11,10,16,28,1 \
    --output-file datasets/aramco_kovacs14_hard.npz