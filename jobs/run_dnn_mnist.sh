#!/bin/bash
#SBATCH --job-name=dnn_mnist
#SBATCH --output=jobs/logs/dnn_mnist_%j.out
#SBATCH --error=jobs/logs/dnn_mnist_%j.err
#SBATCH --partition=ENSTA-l40s
#SBATCH --exclude=ensta-l40s01.r2.enst.fr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_DIR="$HOME/Deep_Learning_2_Project"
cd "$PROJECT_DIR"

#source "$HOME/Deep_Learning_2_Project/venv/bin/activate"
source "$HOME/text-in-image-generation/venv/bin/activate"

mkdir -p jobs/logs outputs

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date   : $(date)"

echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device    : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

python -u principal_DNN_MNIST.py

echo "Done: $(date)"
