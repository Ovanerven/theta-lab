# SingleEnzyme6 — all 6 states observed: A, B, C, D, E, I
# Pairs with SingleEnzymeScaffold (exact Bi-Bi kinetics).
# python last-layer-ode/create_dataset.py --model single_enzyme --t-span 30 --n-steps 200 \
#     --n-samples 1000 --control-indices 0,1 \
#     --obs-indices 0,1,2,3,4,5 \
#     --seed 42 \
#     --output-file datasets/single_enzyme/single_enzyme_6.npz

# python last-layer-ode/create_dataset.py --model single_enzyme --t-span 30 --n-steps 200 \
#     --n-samples 100 --control-indices 0,1 \
#     --obs-indices 0,1,2,3,4,5 \
#     --seed 42 \
#     --output-file datasets/single_enzyme/single_enzyme_6_n100.npz

# python last-layer-ode/create_dataset.py --model single_enzyme --t-span 30 --n-steps 200 \
#     --n-samples 10 --control-indices 0,1 \
#     --obs-indices 0,1,2,3,4,5 \
#     --seed 42 \
#     --output-file datasets/single_enzyme/single_enzyme_6_n10.npz

# python last-layer-ode/create_dataset.py --model single_enzyme --t-span 30 --n-steps 200 \
#     --n-samples 3 --control-indices 0,1 \
#     --obs-indices 0,1,2,3,4,5 \
#     --seed 42 \
#     --output-file datasets/single_enzyme/single_enzyme_6_n3.npz

# SingleEnzymeReduced4 — observe A, B, C, D only.
# Pairs with SingleEnzymeReduced4Scaffold (single_enzyme_4).
python last-layer-ode/create_dataset.py --model single_enzyme --t-span 30 --n-steps 200 \
    --n-samples 1000 --control-indices 0,1 \
    --obs-indices 0,1,2,3 \
    --seed 42 \
    --output-file datasets/single_enzyme/single_enzyme_4.npz

python last-layer-ode/create_dataset.py --model single_enzyme --t-span 30 --n-steps 200 \
    --n-samples 100 --control-indices 0,1 \
    --obs-indices 0,1,2,3 \
    --seed 42 \
    --output-file datasets/single_enzyme/single_enzyme_4_n100.npz

python last-layer-ode/create_dataset.py --model single_enzyme --t-span 30 --n-steps 200 \
    --n-samples 10 --control-indices 0,1 \
    --obs-indices 0,1,2,3 \
    --seed 42 \
    --output-file datasets/single_enzyme/single_enzyme_4_n10.npz

python last-layer-ode/create_dataset.py --model single_enzyme --t-span 30 --n-steps 200 \
    --n-samples 3 --control-indices 0,1 \
    --obs-indices 0,1,2,3 \
    --seed 42 \
    --output-file datasets/single_enzyme/single_enzyme_4_n3.npz

# SingleEnzymeLumped — only A (idx 0) and C (idx 2) observed.
# Pairs with SingleEnzymeLumpedScaffold (2-state first-order, wrong stoichiometry).
# Full 6-state system is still simulated; B, D, E, I are just hidden.
# The scaffold dS=-kf*S+kr*P maps to dA≈..., dC≈..., missing B dependence entirely.
# python last-layer-ode/create_dataset.py --model single_enzyme --t-span 30 --n-steps 200 \
#     --n-samples 1000 --control-indices 0,1 \
#     --obs-indices 0,2 \
#     --seed 42 \
#     --output-file datasets/single_enzyme/single_enzyme_lumped.npz

# python last-layer-ode/create_dataset.py --model single_enzyme --t-span 30 --n-steps 200 \
#     --n-samples 100 --control-indices 0,1 \
#     --obs-indices 0,2 \
#     --seed 42 \
#     --output-file datasets/single_enzyme/single_enzyme_lumped_n100.npz

# python last-layer-ode/create_dataset.py --model single_enzyme --t-span 30 --n-steps 200 \
#     --n-samples 10 --control-indices 0,1 \
#     --obs-indices 0,2 \
#     --seed 42 \
#     --output-file datasets/single_enzyme/single_enzyme_lumped_n10.npz

# python last-layer-ode/create_dataset.py --model single_enzyme --t-span 30 --n-steps 200 \
#     --n-samples 3 --control-indices 0,1 \
#     --obs-indices 0,2 \
#     --seed 42 \
#     --output-file datasets/single_enzyme/single_enzyme_lumped_n3.npz