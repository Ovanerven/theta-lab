# -------------------------
# reduced2: [A, M]
# -------------------------
# python -m src.scripts.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,12" \
#   --obs-indices "0,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "N1000_T300_steps600_zeros_knoise0.0_reduced2_AM.npz"

# -------------------------
# reduced3: [A, J, M]
# -------------------------
# python -m src.scripts.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,9,12" \
#   --obs-indices "0,9,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "N1000_T300_steps600_zeros_knoise0.0_reduced3_AJM.npz"

# -------------------------
# reduced4: [A, G, J, M]
# -------------------------
# python -m src.scripts.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,6,9,12" \
#   --obs-indices "0,6,9,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "N1000_T300_steps600_zeros_knoise0.0_reduced4_AGJM.npz"

# -------------------------
# reduced5: [A, D, G, J, M]
# -------------------------
# python -m src.scripts.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,3,6,9,12" \
#   --obs-indices "0,3,6,9,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "N1000_T300_steps600_zeros_knoise0.0_reduced5_ADGJM.npz"

# -------------------------
# reduced6: [A, B, D, G, J, M]
# -------------------------
# python -m src.scripts.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,1,3,6,9,12" \
#   --obs-indices "0,1,3,6,9,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "N1000_T300_steps600_zeros_knoise0.0_reduced6_ABDGJM.npz"

# -------------------------
# reduced7: [A, D, G, J, K, L, M]
# -------------------------
# python -m src.scripts.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,3,6,9,10,11,12" \
#   --obs-indices "0,3,6,9,10,11,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "N1000_T300_steps600_zeros_knoise0.0_reduced7_ADGJKLM.npz"

# -------------------------
# reduced8: [A, D, G, H, I, J, L, M]
# -------------------------
python -m src.scripts.create_dataset \
  --n-samples 1000 \
  --t-span 300.0 \
  --n-steps 600 \
  --control-indices "0,3,6,7,8,9,11,12" \
  --obs-indices "0,3,6,7,8,9,11,12" \
  --zero-init \
  --k-noise 0.0 \
  --output-file "N1000_T300_steps600_zeros_knoise0.0_8state.npz"

# -------------------------
# reduced9: [A, D, G, H, I, J, K, L, M]
# -------------------------
# python -m src.scripts.create_dataset \
#   --n-samples 1000 \
#   --t-span 300.0 \
#   --n-steps 600 \
#   --control-indices "0,3,6,7,8,9,10,11,12" \
#   --obs-indices "0,3,6,7,8,9,10,11,12" \
#   --zero-init \
#   --k-noise 0.0 \
#   --output-file "N1000_T300_steps600_zeros_knoise0.0_9state.npz"

# -------------------------
# reduced10: [A, B, C, D, E, F, G, H, I, J]
# -------------------------
python -m src.scripts.create_dataset \
  --n-samples 1000 \
  --t-span 300.0 \
  --n-steps 600 \
  --control-indices "0,1,2,3,4,5,6,7,8,9" \
  --obs-indices "0,1,2,3,4,5,6,7,8,9" \
  --zero-init \
  --k-noise 0.0 \
  --output-file "N1000_T300_steps600_zeros_knoise0.0_10state.npz"

# -------------------------
# reduced11: [A, B, C, D, E, F, G, H, I, J, K]
# -------------------------
python -m src.scripts.create_dataset \
  --n-samples 1000 \
  --t-span 300.0 \
  --n-steps 600 \
  --control-indices "0,1,2,3,4,5,6,7,8,9,10" \
  --obs-indices "0,1,2,3,4,5,6,7,8,9,10" \
  --zero-init \
  --k-noise 0.0 \
  --output-file "N1000_T300_steps600_zeros_knoise0.0_11state.npz"

# -------------------------
# reduced12: [A, B, C, D, E, F, G, H, I, J, K, L]
# -------------------------
python -m src.scripts.create_dataset \
  --n-samples 1000 \
  --t-span 300.0 \
  --n-steps 600 \
  --control-indices "0,1,2,3,4,5,6,7,8,9,10,11" \
  --obs-indices "0,1,2,3,4,5,6,7,8,9,10,11" \
  --zero-init \
  --k-noise 0.0 \
  --output-file "N1000_T300_steps600_zeros_knoise0.0_12state.npz"

# -------------------------
# full13: [A, B, C, D, E, F, G, H, I, J, K, L, M]
# -------------------------
python -m src.scripts.create_dataset \
  --n-samples 1000 \
  --t-span 300.0 \
  --n-steps 600 \
  --control-indices "0,1,2,3,4,5,6,7,8,9,10,11,12" \
  --obs-indices "0,1,2,3,4,5,6,7,8,9,10,11,12" \
  --zero-init \
  --k-noise 0.0 \
  --output-file "N1000_T300_steps600_zeros_knoise0.0_13state.npz"