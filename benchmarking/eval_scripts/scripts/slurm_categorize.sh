#!/bin/bash
#SBATCH --job-name=categorize
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/categorize_%j.out
#SBATCH --error=logs/categorize_%j.err

# Usage: sbatch benchmarking/eval_scripts/scripts/slurm_categorize.sh <eval.json> <output.json> <blind|image|report|image_report> [audit_accepted_n]
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
REPO_DIR="/l/users/mashrafi.monon/MICCAI2026-3DMedVLMS/CT-SpatialVQA"
cd "${REPO_DIR}"
mkdir -p logs
set +u
source /apps/local/anaconda3/etc/profile.d/conda.sh
conda activate lingshu-train
set -u
python "${REPO_DIR}/benchmarking/eval_scripts/scripts/categorize_oracle_failures.py" \
    --eval-json "$1" \
    --reports-json "${REPO_DIR}/QA_generation/validation_reports_output.json" \
    --output "$2" \
    --input-mode "$3" \
    --audit-accepted "${4:-500}" \
    --model Qwen/Qwen2.5-14B-Instruct \
    --load-in-4bit \
    --resume
