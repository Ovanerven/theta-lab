#!/bin/bash
set -euo pipefail

# Generate systematic even-size species-labeled datasets that include A and M.
# Full-state indices: A=0,B=1,C=2,D=3,E=4,F=5,G=6,H=7,I=8,J=9,K=10,L=11,M=12

export PYTHONPATH="$(pwd)/last-layer-ode"

# reduced2: [A, M] -> [0,12]
python -m last-layer-ode.create_dataset \
  --n-samples 1000 \
  --t-span 300.0 \
  --n-steps 600 \
  --control-indices "0,12" \
  --obs-indices "0,12" \
  --zero-init \
  --k-noise 0.0 \
  --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced2_AM.npz"

# reduced4_AEIM: [A, E, I, M] -> [0,4,8,12]
python -m last-layer-ode.create_dataset \
  --n-samples 1000 \
  --t-span 300.0 \
  --n-steps 600 \
  --control-indices "0,4,8,12" \
  --obs-indices "0,4,8,12" \
  --zero-init \
  --k-noise 0.0 \
  --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced4_AEIM.npz"

# reduced6_ACFHKM: [A, C, F, H, K, M] -> [0,2,5,7,10,12]
python -m last-layer-ode.create_dataset \
  --n-samples 1000 \
  --t-span 300.0 \
  --n-steps 600 \
  --control-indices "0,2,5,7,10,12" \
  --obs-indices "0,2,5,7,10,12" \
  --zero-init \
  --k-noise 0.0 \
  --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced6_ACFHKM.npz"

# reduced8_ACEGIJLM: [A, C, E, G, I, J, L, M] -> [0,2,4,6,8,9,11,12]
python -m last-layer-ode.create_dataset \
  --n-samples 1000 \
  --t-span 300.0 \
  --n-steps 600 \
  --control-indices "0,2,4,6,8,9,11,12" \
  --obs-indices "0,2,4,6,8,9,11,12" \
  --zero-init \
  --k-noise 0.0 \
  --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced8_ACEGIJLM.npz"

echo "Generated reduced2/reduced4_AEIM/reduced6_ACFHKM/reduced8_ACEGIJLM datasets."
