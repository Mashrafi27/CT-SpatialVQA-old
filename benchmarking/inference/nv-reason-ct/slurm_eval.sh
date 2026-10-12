#!/bin/bash
#SBATCH --job-name=nvreason-eval
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/nvreason_%j.out
#SBATCH --error=logs/nvreason_%j.err

# Usage (from CT-SpatialVQA/):
#   sbatch benchmarking/inference/nv-reason-ct/slurm_eval.sh
# Env overrides: LIMIT (0 = all), ENABLE_THINKING (0/1), MAX_NEW_TOKENS, OUTPUT.
# --resume is always on, so resubmitting continues where a timed-out job stopped.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail

REPO_DIR="/l/users/mashrafi.monon/MICCAI2026-3DMedVLMS/CT-SpatialVQA"
NIFTI_ROOT="/l/users/mashrafi.monon/MICCAI2026-3DMedVLMS/3D_VLM_Spatial/dataset/data_volumes/dataset/valid_fixed"
DATASET="${REPO_DIR}/dataset/ct_spatialvqa/spatial_qa_filtered_full_nifti.jsonl"
OUTPUT="${OUTPUT:-${REPO_DIR}/benchmarking/reports/predictions/nv_reason_ct_predictions_full.jsonl}"

cd "${REPO_DIR}"
mkdir -p logs

# Call the env's interpreter directly; `conda activate` has fallen back to base python here.
PYTHON="/home/mashrafi.monon/.conda/envs/nvreason/bin/python"

THINK_ARGS=""
if [ "${ENABLE_THINKING:-0}" = "1" ]; then
    THINK_ARGS="--enable-thinking"
fi

"${PYTHON}" "${REPO_DIR}/benchmarking/inference/nv-reason-ct/run_custom_eval.py" \
    --dataset "${DATASET}" \
    --nifti-root "${NIFTI_ROOT}" \
    --output "${OUTPUT}" \
    --anatomy-region chest \
    --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
    --limit "${LIMIT:-0}" \
    ${THINK_ARGS} \
    --resume

echo "Evaluation complete. Predictions at ${OUTPUT}"
