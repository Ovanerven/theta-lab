#!/bin/bash
#SBATCH --job-name=smoke_oxy
#SBATCH --output=slurm_outputs/smoke_oxy_%j.out
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=00:10:00
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
  --set study=smoketest_lambda_oxygen \
  --set out_root=/tmp/smoketest_oxy \
  --set exp_name=ox_smoke \
  --set dataset_path=datasets/cell-free/txtl_native_new_only.npz \
  --set scaffold=txtl_resource_and_maturation_dna \
  --set obs_idx='[3, 5]' \
  --set epochs=1 --set warmup_epochs=0 \
  --set hidden=64 --set lift_dim=16 --set num_layers=1 \
  --set val_n=20 --set test_n=20 --set batch_size=16 \
  --set lambda_oxygen=0.5 --set oxygen_protein_channel=5 --set oxygen_u_open_name=u_open \
  --set species_weights='{3: 1.25, 5: 1.0}' \
  --set time_upweight='{5: [0.5, 3.0]}' \
  --set lambda_endpoint=0.1 --set endpoint_channels='[5]' \
  --set loss_normalizer_channels=3 --set loss_clamp_min=1.0 \
  --set stratified_split=true --set stratify_targets='[3, 5]' --set stratify_bins=3 --set stratify_by_source=true \
  --set endpoint_r2=false \
  --no-plot
