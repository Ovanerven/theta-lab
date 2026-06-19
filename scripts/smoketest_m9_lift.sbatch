#!/bin/bash
#SBATCH --job-name=smoke_m9_lift
#SBATCH --output=slurm_outputs/smoke_m9_lift_%j.out
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

module purge
module load 2025
module load Anaconda3/2025.06-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate thesis_env

cd /home/overven1/theta-lab

python -u last-layer-ode/train.py \
  --config configs/txtl_gru_new_baseline.yaml \
  --set study=smoketest_m9_lift \
  --set out_root=/tmp/smoketest_m9_lift \
  --set exp_name=m9_smoke \
  --set scaffold=txtl_model9_oxygen_dark \
  --set dataset_path=datasets/cell-free/txtl_native_real_only.npz \
  --set obs_idx='[3, 5]' \
  --set epochs=1 --set warmup_epochs=0 \
  --set hidden=64 --set lift_dim=16 --set num_layers=1 \
  --set val_n=40 --set test_n=40 --set batch_size=16 \
  --set lambda_oxygen=0.5 --set oxygen_protein_channel=5 --set oxygen_u_open_name=u_open \
  --set loss_clamp_min=1.0 --set n_substeps=2 \
  --set subtract_channel_min=true --set subtract_channel_min_cols='[3, 5]' \
  --set stratified_split=true --set stratify_targets='[3, 5]' --set stratify_bins=3 --set stratify_by_source=true \
  --set endpoint_r2=true \
  --no-plot
