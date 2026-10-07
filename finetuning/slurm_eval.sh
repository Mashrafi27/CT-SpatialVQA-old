#!/bin/bash
#SBATCH --job-name=lingshu-eval
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# ── Paths (update these) ───────────────────────────────────────────
NIFTI_ROOT=""          # e.g. /path/to/valid_fixed
NPZ_ROOT="/l/users/mashrafi.monon/MICCAI2026-3DMedVLMS/3D_VLM_Spatial/preprocess/medevalkit_npz"
# ────────────────────────────────────────────────────────────────────

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail

REPO_DIR="/l/users/mashrafi.monon/MICCAI2026-3DMedVLMS/CT-SpatialVQA"
TEST_JSONL="${REPO_DIR}/finetuning/splits/test.jsonl"

cd "${REPO_DIR}"
mkdir -p logs
ADAPTER_PATH="${ADAPTER_PATH:-${REPO_DIR}/finetuning/checkpoints/lingshu-lora/final}"
OUTPUT="${OUTPUT:-${REPO_DIR}/finetuning/results/lingshu_lora_predictions_test.jsonl}"

set +u  # conda activate scripts reference unset vars
source /apps/local/anaconda3/etc/profile.d/conda.sh
conda activate medevalkit
set -u

pip install peft --quiet 2>/dev/null

VOL_ARGS=""
if [ -n "$NPZ_ROOT" ]; then
    VOL_ARGS="--npz-root ${NPZ_ROOT}"
elif [ -n "$NIFTI_ROOT" ]; then
    VOL_ARGS="--nifti-root ${NIFTI_ROOT}"
else
    echo "ERROR: Set NIFTI_ROOT or NPZ_ROOT at the top of this script."
    exit 1
fi

python "${REPO_DIR}/finetuning/eval_lingshu_lora.py" \
    --test-jsonl "${TEST_JSONL}" \
    ${VOL_ARGS} \
    --adapter-path "${ADAPTER_PATH}" \
    --output "${OUTPUT}" \
    --num-slices 32 \
    --resume

echo "Evaluation complete. Predictions at ${OUTPUT}"
