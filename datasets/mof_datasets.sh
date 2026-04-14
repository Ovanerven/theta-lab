# MOFSynthesis12 — all 12 states
python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 1000 --control-indices 4,5 \
    --obs-indices 0,1,2,3,4,5,6,7,8,9,10,11 \
    --output-file datasets/mof_synthesis_12.npz

# MOFSynthesis8 — Base, Mod, SBU, SBU_capped, Nuc_A, Am, Nuc_C, MOF_C
python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 1000 --control-indices 4,5 \
    --obs-indices 4,5,6,7,8,9,10,11 \
    --output-file datasets/mof_synthesis_8.npz

# MOFSynthesis6 — Base, Mod, SBU, Am, Nuc_C, MOF_C
python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 1000 --control-indices 4,5 \
    --obs-indices 4,5,6,9,10,11 \
    --output-file datasets/mof_synthesis_6.npz

# MOFSynthesis4 — Base, Mod, Am, MOF_C
python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 1000 --control-indices 4,5 \
    --obs-indices 4,5,9,11 \
    --output-file datasets/mof_synthesis_4.npz