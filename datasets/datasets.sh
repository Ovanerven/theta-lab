export PYTHONPATH="$(pwd)/last-layer-ode"
# # Assumes: P_obs == P_control for every dataset.
# # Full-state indices: A=0,B=1,C=2,D=3,E=4,F=5,G=6,H=7,I=8,J=9,K=10,L=11,M=12

# # -------------------------
# # reduced5 (odd): [A, D, G, J, M] -> [0,3,6,9,12]
# # -------------------------
# python -m last-layer-ode.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,3,6,9,12" \
#   --obs-indices "0,3,6,9,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced5_ADGJM.npz"


# # -------------------------
# # full13: [A..M] -> [0..12]
# # -------------------------
# python -m last-layer-ode.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,1,2,3,4,5,6,7,8,9,10,11,12" \
#   --obs-indices "0,1,2,3,4,5,6,7,8,9,10,11,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_full13_ABCDEFGHIJKLM.npz"

# # -------------------------
# # reduced2 (odd): [A, M] -> [0,12]
# # -------------------------
# python -m last-layer-ode.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,12" \
#   --obs-indices "0,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced2_AM.npz"

# # -------------------------
# # reduced3 (odd): [A, J, M] -> [0,9,12]
# # -------------------------
# python -m last-layer-ode.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,9,12" \
#   --obs-indices "0,9,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced3_AJM.npz"

# # -------------------------
# # reduced4 (even noM): [A, G, J, L] -> [0,6,9,11]
# # -------------------------
# python -m last-layer-ode.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,6,9,11" \
#   --obs-indices "0,6,9,11" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced4_AGJL.npz"

# # -------------------------
# # reduced6 (even noM): [A, B, D, G, J, L] -> [0,1,3,6,9,11]
# # -------------------------
# python -m last-layer-ode.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,1,3,6,9,11" \
#   --obs-indices "0,1,3,6,9,11" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced6_ABDGJL.npz"

# # -------------------------
# # reduced7 (odd): [A, D, G, J, K, L, M] -> [0,3,6,9,10,11,12]
# # -------------------------
# python -m last-layer-ode.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,3,6,9,10,11,12" \
#   --obs-indices "0,3,6,9,10,11,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced7_ADGJKLM.npz"

# # -------------------------
# # reduced8 (even noM): [A, D, G, H, I, J, K, L] -> [0,3,6,7,8,9,10,11]
# # -------------------------
# python -m last-layer-ode.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,3,6,7,8,9,10,11" \
#   --obs-indices "0,3,6,7,8,9,10,11" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced8_ADGHIJKL.npz"


# -------------------------
# reduced8 (even noM): [A, D, G, H, J, K, L, M] -> [0,3,6,7,8,9,10,11]
# -------------------------
# python -m last-layer-ode.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,3,6,7,9,10,11,12" \
#   --obs-indices "0,3,6,7,9,10,11,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced8_ADGHJKLM.npz"

# # -------------------------
# # reduced9 (odd): [A, D, G, H, I, J, K, L, M] -> [0,3,6,7,8,9,10,11,12]
# # -------------------------
# python -m last-layer-ode.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,3,6,7,8,9,10,11,12" \
#   --obs-indices "0,3,6,7,8,9,10,11,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced9_ADGHIJKLM.npz"

# # -------------------------
# # reduced10 (even noM): [A, B, C, D, E, F, G, H, I, L] -> [0,1,2,3,4,5,6,7,8,11]
# # -------------------------
# python -m last-layer-ode.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,1,2,3,4,5,6,7,8,11" \
#   --obs-indices "0,1,2,3,4,5,6,7,8,11" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced10_ABCDEFGHIL.npz"

# # -------------------------
# # reduced11 (odd): [A, B, C, D, E, F, G, H, I, J, M] -> [0,1,2,3,4,5,6,7,8,9,12]
# # -------------------------
# python -m last-layer-ode.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,1,2,3,4,5,6,7,8,9,12" \
#   --obs-indices "0,1,2,3,4,5,6,7,8,9,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced11_ABCDEFGHIJM.npz"

# # -------------------------
# # reduced12 (even): [A, B, C, D, E, F, G, H, I, J, K, L] -> [0,1,2,3,4,5,6,7,8,9,10,11]
# # -------------------------
# python -m last-layer-ode.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,1,2,3,4,5,6,7,8,9,10,11" \
#   --obs-indices "0,1,2,3,4,5,6,7,8,9,10,11" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced12_ABCDEFGHIJKL.npz"

# # ------------------------------------------------------------
# # reduced6_ADGJLM: (A, D, G, J, L, M) -> [0,3,6,9,11,12]
# # ------------------------------------------------------------
# python -m last-layer-ode.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,3,6,9,11,12" \
#   --obs-indices "0,3,6,9,11,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced6_ADGJLM.npz"

# # ------------------------------------------------------------
# # reduced6_AGHILM: (A, G, H, I, L, M) -> [0,6,7,8,11,12]
# # ------------------------------------------------------------
# python -m last-layer-ode.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,6,7,8,11,12" \
#   --obs-indices "0,6,7,8,11,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced6_AGHILM.npz"

# # ------------------------------------------------------------
# # reduced6_DGHILM: (D, G, H, I, L, M) -> [3,6,7,8,11,12]
# # ------------------------------------------------------------
# python -m last-layer-ode.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "3,6,7,8,11,12" \
#   --obs-indices "3,6,7,8,11,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced6_DGHILM.npz"

# # ------------------------------------------------------------
# # reduced6_ABCDLM: (A, B, C, D, L, M) -> [0,1,2,3,11,12]
# # ------------------------------------------------------------
# python -m last-layer-ode.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,1,2,3,11,12" \
#   --obs-indices "0,1,2,3,11,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced6_ABCDLM.npz"

# python last-layer-ode/create_dataset.py \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,1,2,3,4,5,6,7,8,9,10,11,12" \
#   --obs-indices "0,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file datasets/N1000_T300_steps600_zeros_knoise0.0_obs_AM_control_full13

# reduced10_with_M: [A,D,F,G,H,I,J,K,L,M] = indices 0,3,5,6,7,8,9,10,11,12
python -m last-layer-ode.create_dataset \
  --n-samples 1000 --t-span 300.0 --n-steps 600 \
  --control-indices "0,3,5,6,7,8,9,10,11,12" \
  --obs-indices     "0,3,5,6,7,8,9,10,11,12" \
  --zero-init --k-noise 0.0 \
  --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced10_ADFGHIJKLM.npz"

# reduced11_with_M: [A,D,E,F,G,H,I,J,K,L,M] = indices 0,3,4,5,6,7,8,9,10,11,12
python -m last-layer-ode.create_dataset \
  --n-samples 1000 --t-span 300.0 --n-steps 600 \
  --control-indices "0,3,4,5,6,7,8,9,10,11,12" \
  --obs-indices     "0,3,4,5,6,7,8,9,10,11,12" \
  --zero-init --k-noise 0.0 \
  --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced11_ADEFGHIJKLM.npz"

# reduced12_with_M: [A,B,D,E,F,G,H,I,J,K,L,M] = indices 0,1,3,4,5,6,7,8,9,10,11,12
python -m last-layer-ode.create_dataset \
  --n-samples 1000 --t-span 300.0 --n-steps 600 \
  --control-indices "0,1,3,4,5,6,7,8,9,10,11,12" \
  --obs-indices     "0,1,3,4,5,6,7,8,9,10,11,12" \
  --zero-init --k-noise 0.0 \
  --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced12_ABDEFGHIJKLM.npz"

python -u last-layer-ode/baselines/honest_rollout.py \     
--dataset-dir datasets \
--scaffolds reduced2,reduced3,reduced4,reduced5,reduced6_ADGJLM,reduced7,reduced8_ADGHJKLM,reduced9,reduced10_with_M,reduced11_with_M,reduced12_with_M,full13 \
--sample-idx 0 \
--gd-steps 400 \                
--out results/honest_rollout.png