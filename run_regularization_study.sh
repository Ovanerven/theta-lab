#!/usr/bin/env bash
# Regularization study: baseline + L1/L2 (λ=0.01,0.1,1.0) across all scaffold sizes (P=2..13)
# Run from repo root: bash run_regularization_study.sh
set -e
cd "$(dirname "$0")"

CONFIGS=configs/regularization_study

# ── 12 models × 7 variants = 84 runs ─────────────────────────────────────

# reduced2
python last-layer-ode/train.py --config $CONFIGS/reduced2/baseline.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced2/l1_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced2/l1_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced2/l1_1.0.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced2/l2_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced2/l2_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced2/l2_1.0.yaml

# reduced3
python last-layer-ode/train.py --config $CONFIGS/reduced3/baseline.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced3/l1_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced3/l1_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced3/l1_1.0.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced3/l2_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced3/l2_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced3/l2_1.0.yaml

# reduced4
python last-layer-ode/train.py --config $CONFIGS/reduced4/baseline.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced4/l1_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced4/l1_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced4/l1_1.0.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced4/l2_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced4/l2_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced4/l2_1.0.yaml

# reduced5
python last-layer-ode/train.py --config $CONFIGS/reduced5/baseline.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced5/l1_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced5/l1_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced5/l1_1.0.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced5/l2_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced5/l2_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced5/l2_1.0.yaml

# reduced6
python last-layer-ode/train.py --config $CONFIGS/reduced6/baseline.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced6/l1_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced6/l1_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced6/l1_1.0.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced6/l2_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced6/l2_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced6/l2_1.0.yaml

# reduced7
python last-layer-ode/train.py --config $CONFIGS/reduced7/baseline.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced7/l1_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced7/l1_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced7/l1_1.0.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced7/l2_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced7/l2_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced7/l2_1.0.yaml

# reduced8
python last-layer-ode/train.py --config $CONFIGS/reduced8/baseline.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced8/l1_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced8/l1_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced8/l1_1.0.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced8/l2_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced8/l2_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced8/l2_1.0.yaml

# reduced9
python last-layer-ode/train.py --config $CONFIGS/reduced9/baseline.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced9/l1_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced9/l1_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced9/l1_1.0.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced9/l2_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced9/l2_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced9/l2_1.0.yaml

# reduced10
python last-layer-ode/train.py --config $CONFIGS/reduced10/baseline.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced10/l1_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced10/l1_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced10/l1_1.0.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced10/l2_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced10/l2_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced10/l2_1.0.yaml

# reduced11
python last-layer-ode/train.py --config $CONFIGS/reduced11/baseline.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced11/l1_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced11/l1_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced11/l1_1.0.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced11/l2_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced11/l2_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced11/l2_1.0.yaml

# reduced12
python last-layer-ode/train.py --config $CONFIGS/reduced12/baseline.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced12/l1_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced12/l1_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced12/l1_1.0.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced12/l2_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced12/l2_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/reduced12/l2_1.0.yaml

# full13
python last-layer-ode/train.py --config $CONFIGS/full13/baseline.yaml
python last-layer-ode/train.py --config $CONFIGS/full13/l1_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/full13/l1_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/full13/l1_1.0.yaml
python last-layer-ode/train.py --config $CONFIGS/full13/l2_0.01.yaml
python last-layer-ode/train.py --config $CONFIGS/full13/l2_0.1.yaml
python last-layer-ode/train.py --config $CONFIGS/full13/l2_1.0.yaml

echo "=== Regularization study complete. Results in experiments/regularization_study/ ==="
