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
To train the model (basic GRU), run:
```sh
python -m src.train.train --data datasets/N1000_T300_steps600_zeros_knoise0.0.npz --epochs 500 --exp-name "log10_loss_no_clampmin_500epoch" --checkpoint-every 50
```

## Requirements
- LaTeX
- BibTeX

## License
Specify the license here.