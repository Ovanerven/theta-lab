# Thesis

## Overview
Brief description of the thesis and its goals.

## Structure
- `chapter-1/`
- `chapter-2/`
- `appendix/`

## Build
```sh
# example build command
make pdf
```

## Dataset Generation
To generate the dataset, run:
```sh
python -m src.scripts.create_dataset --n-samples 10 --t-span 300 --n-steps 600 --tail 0.0 --seed 42 --zero-init
```
This will create the output file `dataset_N10_T300_K600_zero_tail0_seed42.npz`.

## Model Training

Experiment configs live in `configs/`. Outputs (model, logs, plots) go into `experiments/{exp_name}/`.

```sh
# Run an experiment from its YAML config
python -m src.scripts.train --config configs/full13_obs13.yaml

# Re-plot an existing experiment
python -m src.scripts.plot_all --exp-name full13_obs13
```

To add a new experiment, copy an existing config and change `output.exp_name` and any hyperparameters:
```sh
cp configs/full13_obs13.yaml configs/my_new_experiment.yaml
# edit configs/my_new_experiment.yaml
python -m src.scripts.train --config configs/my_new_experiment.yaml
```

## Requirements
- LaTeX
- BibTeX

## License
Specify the license here.