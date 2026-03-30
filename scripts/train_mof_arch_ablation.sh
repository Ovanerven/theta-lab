#!/usr/bin/env bash
# Architecture ablation: GRU vs Transformer vs MLP, across MOF scaffold sizes 4/6/8/12.
# Results land in experiments/mof_arch_ablation/



set -e

TRAIN="python last-layer-ode/train.py"

# --- GRU (baseline) ---
$TRAIN --config configs/mof_arch_ablation/mof4_gru.yaml
$TRAIN --config configs/mof_arch_ablation/mof6_gru.yaml
$TRAIN --config configs/mof_arch_ablation/mof8_gru.yaml
$TRAIN --config configs/mof_arch_ablation/mof12_gru.yaml

# --- Transformer (context_len=64, batch_size=32) ---
$TRAIN --config configs/mof_arch_ablation/mof4_transformer.yaml
$TRAIN --config configs/mof_arch_ablation/mof6_transformer.yaml
$TRAIN --config configs/mof_arch_ablation/mof8_transformer.yaml
$TRAIN --config configs/mof_arch_ablation/mof12_transformer.yaml

# --- MLP / Markovian (no history) ---
$TRAIN --config configs/mof_arch_ablation/mof4_mlp.yaml
$TRAIN --config configs/mof_arch_ablation/mof6_mlp.yaml
$TRAIN --config configs/mof_arch_ablation/mof8_mlp.yaml
$TRAIN --config configs/mof_arch_ablation/mof12_mlp.yaml
