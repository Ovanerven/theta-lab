# MOFSynthesis12 — all 12 states
python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 1000 --control-indices 4,5 \
    --obs-indices 0,1,2,3,4,5,6,7,8,9,10,11 \
    --output-file datasets/mof/mof_synthesis_12_n1000.npz

python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 3 --control-indices 4,5 \
    --obs-indices 0,1,2,3,4,5,6,7,8,9,10,11 \
    --output-file datasets/mof/mof_synthesis_12_n3.npz

python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 100 --control-indices 4,5 \
    --obs-indices 0,1,2,3,4,5,6,7,8,9,10,11 \
    --output-file datasets/mof/mof_synthesis_12_n100.npz

python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 10 --control-indices 4,5 \
    --obs-indices 0,1,2,3,4,5,6,7,8,9,10,11 \
    --output-file datasets/mof/mof_synthesis_12_n10.npz

# MOFSynthesis8 — Base, Mod, SBU, SBU_capped, Nuc_A, Am, Nuc_C, MOF_C
python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 1000 --control-indices 4,5 \
    --obs-indices 4,5,6,7,8,9,10,11 \
    --output-file datasets/mof/mof_synthesis_8_n1000.npz

python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 3 --control-indices 4,5 \
    --obs-indices 4,5,6,7,8,9,10,11 \
    --output-file datasets/mof/mof_synthesis_8_n3.npz

python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 100 --control-indices 4,5 \
    --obs-indices 4,5,6,7,8,9,10,11 \
    --output-file datasets/mof/mof_synthesis_8_n100.npz

python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 10 --control-indices 4,5 \
    --obs-indices 4,5,6,7,8,9,10,11 \
    --output-file datasets/mof/mof_synthesis_8_n10.npz

# MOFSynthesis6 — Base, Mod, SBU, Am, Nuc_C, MOF_C
python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 1000 --control-indices 4,5 \
    --obs-indices 4,5,6,9,10,11 \
    --output-file datasets/mof/mof_synthesis_6_n1000.npz

python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 3 --control-indices 4,5 \
    --obs-indices 4,5,6,9,10,11 \
    --output-file datasets/mof/mof_synthesis_6_n3.npz

python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 100 --control-indices 4,5 \
    --obs-indices 4,5,6,9,10,11 \
    --output-file datasets/mof/mof_synthesis_6_n100.npz

python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 10 --control-indices 4,5 \
    --obs-indices 4,5,6,9,10,11 \
    --output-file datasets/mof/mof_synthesis_6_n10.npz

# MOFSynthesis4 — Base, Mod, Am, MOF_C
python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 1000 --control-indices 4,5 \
    --obs-indices 4,5,9,11 \
    --output-file datasets/mof/mof_synthesis_4_n1000.npz

python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 3 --control-indices 4,5 \
    --obs-indices 4,5,9,11 \
    --output-file datasets/mof/mof_synthesis_4_n3.npz
    
# MOFSynthesis4 — Base, Mod, Am, MOF_C
python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 100 --control-indices 4,5 \
    --obs-indices 4,5,9,11 \
    --output-file datasets/mof/mof_synthesis_4_n100.npz

python last-layer-ode/create_dataset.py --model mof_synthesis --t-span 30 --n-steps 300 \
    --n-samples 10 --control-indices 4,5 \
    --obs-indices 4,5,9,11 \
    --output-file datasets/mof/mof_synthesis_4_n10.npz