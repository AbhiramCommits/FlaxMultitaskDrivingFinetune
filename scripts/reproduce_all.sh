#!/usr/bin/env bash
set -e

echo "=== FlaxMultitaskDrivingFinetune End-to-End Reproduction Script ==="

# 1. Data generation & SQL windowing
python -m fmdf.data.generate --n-logs 1200 --out data/raw.parquet --seed 42
python -m fmdf.data.sql --build

# 2. Pretraining
python -m fmdf.train.pretrain --config configs/base.yaml --steps 500

# 3. Fine-tuning arms
python -m fmdf.train.finetune --mode lora --steps 100
python -m fmdf.train.finetune --mode full --steps 100
python -m fmdf.train.finetune --mode head_only --steps 100

# 4. Scaling, Sweep, Domain Shift, Sensitivity & Report
python -m fmdf.experiments.scaling
python -m fmdf.experiments.sweep_loss_weights
python -m fmdf.experiments.domain_shift
python -m fmdf.eval.sensitivity
python -m fmdf.eval.report

echo "=== End-to-End Reproduction Completed Successfully ==="
