#!/bin/bash
#SBATCH --job-name=rbm_alpha
#SBATCH --output=jobs/logs/rbm_alpha_%j.out
#SBATCH --error=jobs/logs/rbm_alpha_%j.err
#SBATCH --partition=ENSTA-l40s
#SBATCH --exclude=ensta-l40s01.r2.enst.fr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00

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

python -u principal_RBM_alpha.py

echo "Done: $(date)"
