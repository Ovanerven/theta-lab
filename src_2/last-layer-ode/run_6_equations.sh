# ------------------------------------------------------------
# reduced6_ADGJLM: (A, D, G, J, L, M) -> [0,3,6,9,11,12]
# ------------------------------------------------------------
python -m last-layer-ode.create_dataset \
  --n-samples 1000 \
  --t-span 300.0 \
  --n-steps 600 \
  --control-indices "0,3,6,9,11,12" \
  --obs-indices "0,3,6,9,11,12" \
  --zero-init \
  --k-noise 0.0 \
  --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced6_ADGJLM.npz"

# ------------------------------------------------------------
# reduced6_AGHILM: (A, G, H, I, L, M) -> [0,6,7,8,11,12]
# ------------------------------------------------------------
python -m last-layer-ode.create_dataset \
  --n-samples 1000 \
  --t-span 300.0 \
  --n-steps 600 \
  --control-indices "0,6,7,8,11,12" \
  --obs-indices "0,6,7,8,11,12" \
  --zero-init \
  --k-noise 0.0 \
  --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced6_AGHILM.npz"

# ------------------------------------------------------------
# reduced6_DGHILM: (D, G, H, I, L, M) -> [3,6,7,8,11,12]
# ------------------------------------------------------------
python -m last-layer-ode.create_dataset \
  --n-samples 1000 \
  --t-span 300.0 \
  --n-steps 600 \
  --control-indices "3,6,7,8,11,12" \
  --obs-indices "3,6,7,8,11,12" \
  --zero-init \
  --k-noise 0.0 \
  --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced6_DGHILM.npz"

# ------------------------------------------------------------
# reduced6_ABCDLM: (A, B, C, D, L, M) -> [0,1,2,3,11,12]
# ------------------------------------------------------------
python -m last-layer-ode.create_dataset \
  --n-samples 1000 \
  --t-span 300.0 \
  --n-steps 600 \
  --control-indices "0,1,2,3,11,12" \
  --obs-indices "0,1,2,3,11,12" \
  --zero-init \
  --k-noise 0.0 \
  --output-file "datasets/N1000_T300_steps600_zeros_knoise0.0_reduced6_ABCDLM.npz"

python train.py --config configs/reduced6_ADGJLM.yaml
python train.py --config configs/reduced6_AGHILM.yaml
python train.py --config configs/reduced6_DGHILM.yaml
python train.py --config configs/reduced6_ABCDLM.yaml