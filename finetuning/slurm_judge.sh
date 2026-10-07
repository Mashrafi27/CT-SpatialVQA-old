#!/bin/bash
#SBATCH --job-name=qwen-judge
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/judge_%j.out
#SBATCH --error=logs/judge_%j.err

# Usage: sbatch finetuning/slurm_judge.sh <predictions.jsonl> <output.json> [<predictions.jsonl> <output.json> ...]
# Judges each predictions file in turn (model loads once per file; --resume skips judged items).

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail

REPO_DIR="/l/users/mashrafi.monon/MICCAI2026-3DMedVLMS/CT-SpatialVQA"
cd "${REPO_DIR}"
mkdir -p logs

set +u  # conda activate scripts reference unset vars
source /apps/local/anaconda3/etc/profile.d/conda.sh
conda activate lingshu-train
set -u

while [ $# -ge 2 ]; do
    python "${REPO_DIR}/finetuning/judge_qwen_local.py" \
        --predictions "$1" \
        --output "$2" \
        --model Qwen/Qwen2.5-14B-Instruct \
        --load-in-4bit \
        --resume
    shift 2
done
