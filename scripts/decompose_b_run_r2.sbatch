#!/bin/bash
#SBATCH --job-name=decompose_b_r2
#SBATCH --output=slurm_outputs/decompose_b_r2_%j.out
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

module purge
module load 2025
module load Anaconda3/2025.06-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate thesis_env

cd /home/overven/theta-lab
python scripts/decompose_b_run_r2.py
