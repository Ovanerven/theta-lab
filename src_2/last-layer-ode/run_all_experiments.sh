#!/bin/bash

# Run all reduced model experiments for new codebase

# python train.py --config configs/reduced3.yaml
# python train.py --config configs/reduced7.yaml
# python train.py --config configs/reduced9.yaml
# python train.py --config configs/reduced11.yaml
# python train.py --config configs/reduced13.yaml
python train.py --config configs/reduced6.yaml
python train.py --config configs/reduced4.yaml
python train.py --config configs/reduced8.yaml
python train.py --config configs/reduced5_2layer.yaml
python train.py --config configs/reduced5.yaml
# python train.py --config configs/reduced5_nstep_20.yaml
python train.py --config configs/reduced7_2layer.yaml
python train.py --config configs/reduced7_nstep_20.yaml

python train.py --config configs/reduced10.yaml
python train.py --config configs/reduced12.yaml

python train.py --config configs/reduced3_2layer.yaml
python train.py --config configs/reduced3_nstep_20.yaml

# python train.py --config configs/reduced9_2layer.yaml
# python train.py --config configs/reduced9_nstep_20.yaml
# python train.py --config configs/reduced11_2layer.yaml
# python train.py --config configs/reduced11_nstep_20.yaml
# python train.py --config configs/reduced13_2layer.yaml
# python train.py --config configs/reduced13_nstep_20.yaml


