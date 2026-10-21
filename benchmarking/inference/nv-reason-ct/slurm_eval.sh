#!/bin/bash
#SBATCH --job-name=nvreason-eval
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/nvreason_%A_%a.out
#SBATCH --error=logs/nvreason_%A_%a.err

# Usage (from CT-SpatialVQA/):
#   sbatch benchmarking/inference/nv-reason-ct/slurm_eval.sh                      # single job
#   sbatch --array=0-5 benchmarking/inference/nv-reason-ct/slurm_eval.sh          # 6 parallel shards
# Env overrides: LIMIT (0 = all), ENABLE_THINKING (0/1), MAX_NEW_TOKENS, OUTPUT,
#   INPUT_MODE (image | blind | report | image_report), REPORTS_JSON.
# Ablations:
#   INPUT_MODE=blind  OUTPUT=.../nv_reason_ct_blind_predictions_full.jsonl sbatch --array=0-2 ...
#   INPUT_MODE=report OUTPUT=.../nv_reason_ct_report_predictions_full.jsonl sbatch --array=0-2 ...
#   INPUT_MODE=image_report OUTPUT=.../nv_reason_ct_image_report_predictions_full.jsonl sbatch --array=0-5 ...
# As an array job each task handles one contiguous shard and writes OUTPUT with a
# ".shard<i>of<n>" suffix; merge afterwards with:
#   cat <OUTPUT>.shard*of<n> > <OUTPUT>
# --resume is always on, so resubmitting continues where a timed-out job stopped.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Compute nodes cannot reach the HF Hub; the model is cached, so stay offline
# (otherwise from_pretrained hangs on a CDN request before loading weights).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
set -euo pipefail

REPO_DIR="/l/users/mashrafi.monon/MICCAI2026-3DMedVLMS/CT-SpatialVQA"
NIFTI_ROOT="/l/users/mashrafi.monon/MICCAI2026-3DMedVLMS/3D_VLM_Spatial/dataset/data_volumes/dataset/valid_fixed"
DATASET="${REPO_DIR}/dataset/ct_spatialvqa/spatial_qa_filtered_full_nifti.jsonl"
OUTPUT="${OUTPUT:-${REPO_DIR}/benchmarking/reports/predictions/nv_reason_ct_predictions_full.jsonl}"
INPUT_MODE="${INPUT_MODE:-image}"
REPORTS_JSON="${REPORTS_JSON:-${REPO_DIR}/QA_generation/validation_reports_output.json}"

cd "${REPO_DIR}"
mkdir -p logs

# Call the env's interpreter directly; `conda activate` has fallen back to base python here.
PYTHON="/home/mashrafi.monon/.conda/envs/nvreason/bin/python"

THINK_ARGS=""
if [ "${ENABLE_THINKING:-0}" = "1" ]; then
    THINK_ARGS="--enable-thinking"
fi

SHARD_ARGS=""
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] && [ "${SLURM_ARRAY_TASK_COUNT}" -gt 1 ]; then
    SHARD_ARGS="--num-shards ${SLURM_ARRAY_TASK_COUNT} --shard-index ${SLURM_ARRAY_TASK_ID}"
    OUTPUT="${OUTPUT}.shard${SLURM_ARRAY_TASK_ID}of${SLURM_ARRAY_TASK_COUNT}"
fi

"${PYTHON}" "${REPO_DIR}/benchmarking/inference/nv-reason-ct/run_custom_eval.py" \
    --dataset "${DATASET}" \
    --nifti-root "${NIFTI_ROOT}" \
    --output "${OUTPUT}" \
    --anatomy-region chest \
    --input-mode "${INPUT_MODE}" \
    --reports-json "${REPORTS_JSON}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
    --limit "${LIMIT:-0}" \
    ${THINK_ARGS} \
    ${SHARD_ARGS} \
    --resume

echo "Evaluation complete. Predictions at ${OUTPUT}"
