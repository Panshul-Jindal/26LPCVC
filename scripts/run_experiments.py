import os

configs = [
    "configs/resnet_base.yaml",
    "configs/resnet_lr001.yaml",
    "configs/resnet_unfreeze.yaml",
]

for cfg in configs:
    os.system(f"python train.py --config {cfg}")