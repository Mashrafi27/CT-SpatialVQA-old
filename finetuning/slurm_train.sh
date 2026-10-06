#!/bin/bash
#SBATCH --job-name=lingshu-lora
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ── Paths (update these) ───────────────────────────────────────────
NIFTI_ROOT=""          # e.g. /path/to/valid_fixed
NPZ_ROOT="/l/users/mashrafi.monon/MICCAI2026-3DMedVLMS/3D_VLM_Spatial/preprocess/medevalkit_npz"
# ────────────────────────────────────────────────────────────────────

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail

REPO_DIR="/l/users/mashrafi.monon/MICCAI2026-3DMedVLMS/CT-SpatialVQA"
TRAIN_JSONL="${REPO_DIR}/finetuning/splits/train.jsonl"
OUTPUT_DIR="${REPO_DIR}/finetuning/checkpoints/lingshu-lora"

cd "${REPO_DIR}"
mkdir -p logs

source activate medevalkit 2>/dev/null || conda activate medevalkit

# Install fine-tuning deps if missing
pip install peft bitsandbytes --quiet 2>/dev/null

# Build volume args
VOL_ARGS=""
if [ -n "$NPZ_ROOT" ]; then
    VOL_ARGS="--npz-root ${NPZ_ROOT}"
elif [ -n "$NIFTI_ROOT" ]; then
    VOL_ARGS="--nifti-root ${NIFTI_ROOT}"
else
    echo "ERROR: Set NIFTI_ROOT or NPZ_ROOT at the top of this script."
    exit 1
fi

python "${REPO_DIR}/finetuning/train_lingshu_lora.py" \
    --train-jsonl "${TRAIN_JSONL}" \
    ${VOL_ARGS} \
    --output-dir "${OUTPUT_DIR}" \
    --num-slices 4 \
    --epochs 3 \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --lr 2e-4 \
    --lora-r 16 \
    --lora-alpha 32 \
    --save-steps 200 \
    --logging-steps 10

echo "Training complete. Adapter saved to ${OUTPUT_DIR}/final"
