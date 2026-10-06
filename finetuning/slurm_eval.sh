#!/bin/bash
#SBATCH --job-name=lingshu-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# ── Paths (update these) ───────────────────────────────────────────
NIFTI_ROOT=""          # e.g. /path/to/valid_fixed
NPZ_ROOT=""            # e.g. /path/to/medevalkit_npz (if preprocessed)
# ────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEST_JSONL="${REPO_DIR}/finetuning/splits/test.jsonl"
ADAPTER_PATH="${REPO_DIR}/finetuning/checkpoints/lingshu-lora/final"
OUTPUT="${REPO_DIR}/finetuning/results/lingshu_lora_predictions_test.jsonl"

source activate medevalkit 2>/dev/null || conda activate medevalkit

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
