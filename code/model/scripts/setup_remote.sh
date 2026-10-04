#!/bin/bash
# Remote environment setup for AutoDL RTX 5090
set -e
source /etc/profile

echo "=== Setting up Phase 2 environment ==="

# Install Python dependencies (torch already installed)
python -m pip install --upgrade pip
python -m pip install \
    transformers>=4.45.0 \
    accelerate>=1.0.0 \
    peft>=0.13.0 \
    safetensors>=0.4.0 \
    datasets>=3.0.0 \
    pandas>=2.2.0 \
    numpy>=1.26.0 \
    pyarrow>=15.0.0 \
    scikit-learn>=1.5.0 \
    xgboost>=2.1.0 \
    ta>=0.11.0 \
    wandb>=0.18.0 \
    loguru>=0.7.0 \
    tensorboard>=2.18.0 \
    pyyaml>=6.0 \
    tqdm>=4.66.0 \
    matplotlib>=3.9.0 \
    seaborn>=0.13.0

echo "=== Creating project directories ==="
mkdir -p /root/project/data/raw/crypto
mkdir -p /root/project/outputs/pretrain

echo "=== Verifying installation ==="
python -c "
import torch
import transformers
import pandas
import sklearn
import xgboost
import ta
import loguru
import yaml
print(f'PyTorch:       {torch.__version__}')
print(f'CUDA:          {torch.cuda.is_available()} / {torch.cuda.get_device_name(0)}')
print(f'Transformers:  {transformers.__version__}')
print(f'scikit-learn:  {sklearn.__version__}')
print(f'xgboost:       {xgboost.__version__}')
print('All packages OK')
"

echo "=== Setup complete ==="
