#!/bin/bash
set -e

# Setup environment details (wandb project, etc.)
export WANDB_PROJECT="lpcv2026"

# Print details
echo "Starting Training Run with Differential LR using Adam Optimizer"
echo "Config: configs/resnet_freezeL3L4_adam_cosine_lr_0001_lr_fc_001.yaml"
echo "--------------------------------------------------------------"

# Run the PyTorch training script
# Submitting with 1 GPU initially to test functionality
python src/train.py --config configs/resnet_freezeL3L4_adam_cosine_lr_0001_lr_fc_001.yaml

echo "--------------------------------------------------------------"
echo "Training finished successfully!"
