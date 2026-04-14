# SingleEnzyme6 — all 6 states: A, B, C, D, E, I
python last-layer-ode/create_dataset.py --model single_enzyme --t-span 10 --n-steps 200 \
    --n-samples 2000 --control-indices 0,1 \
    --obs-indices 0,1,2,3,4,5 \
    --seed 42 \
    --output-file datasets/single_enzyme_6.npz

# # SingleEnzyme4 — A, B, C, D (drop inert E and I)
# python last-layer-ode/create_dataset.py --model single_enzyme --t-span 10 --n-steps 200 \
#     --n-samples 2000 --control-indices 0,1 \
#     --obs-indices 0,1,2,3 \
#     --seed 42 \
#     --output-file datasets/single_enzyme_4.npz

# # SingleEnzyme2 — A, C (one substrate + one product, minimal observable)
# python last-layer-ode/create_dataset.py --model single_enzyme --t-span 10 --n-steps 200 \
#     --n-samples 2000 --control-indices 0,1 \
#     --obs-indices 0,2 \
#     --seed 42 \
#     --output-file datasets/single_enzyme_2.npz
