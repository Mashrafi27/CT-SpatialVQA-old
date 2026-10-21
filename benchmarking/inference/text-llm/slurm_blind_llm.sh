#!/bin/bash
#SBATCH --job-name=blind-llm
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/blind_llm_%j.out
#SBATCH --error=logs/blind_llm_%j.err

# Usage: sbatch benchmarking/inference/text-llm/slurm_blind_llm.sh
# Env overrides: MODEL (HF id), OUTPUT, LIMIT.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
REPO_DIR="/l/users/mashrafi.monon/MICCAI2026-3DMedVLMS/CT-SpatialVQA"
cd "${REPO_DIR}"
mkdir -p logs
set +u
source /apps/local/anaconda3/etc/profile.d/conda.sh
conda activate lingshu-train
set -u
MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
OUTPUT="${OUTPUT:-${REPO_DIR}/benchmarking/reports/predictions/qwen2.5_14b_blind_predictions_full.jsonl}"
python "${REPO_DIR}/benchmarking/inference/text-llm/run_blind_llm.py" \
    --dataset "${REPO_DIR}/dataset/ct_spatialvqa/spatial_qa_filtered_full_nifti.jsonl" \
    --output "${OUTPUT}" \
    --model "${MODEL}" \
    --load-in-4bit \
    --limit "${LIMIT:-0}" \
    --resume
echo "Done. Predictions at ${OUTPUT}"
