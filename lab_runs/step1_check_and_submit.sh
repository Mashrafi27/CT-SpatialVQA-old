#!/bin/bash
# Run on the lab, from the repo root:  bash lab_runs/step1_check_and_submit.sh
# 1 kills the leftover login-node lens process (readout_axes.py lens) on this host, by PID
# 2 submits the per-axis readout job (SLURM CPU job; never runs analysis here)
# 3 records what CT-RATE data / disk space exist for the fine-tuning pilot (task 7)
# 4 copies NV-Reason-CT's small code/config files (no weights) for writing the training script
# Then commits lab_runs/ and pushes, so the results can be read off GitHub.
set -u
cd "$(dirname "$0")/.."
D=$(date +%Y-%m-%d_%H%M)
mkdir -p lab_runs/status lab_runs/nvreason_model_code
OUT=lab_runs/status/${D}_step1.txt
A=benchmarking/reports/analysis
U=/l/users/mashrafi.monon

main() {
  echo "host $(hostname)  date $(date)  user $(whoami)  branch $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"

  echo; echo "== 1 leftover lens processes on this host"
  found=0
  for pid in $(pgrep -u "$(whoami)" -f "readout_axes.py lens"); do
    found=1
    ps -o pid,etime,pcpu,rss,args -p "$pid"
    kill "$pid" && echo "killed $pid"
  done
  [ $found = 0 ] && echo "none on $(hostname) (if this is not the login node, the login node was not checked)"

  echo; echo "== 2 lens output"
  ls -la $A/readout_all_axis_lens.npz 2>/dev/null || echo "missing; the SLURM job computes it first"

  echo; echo "== 3 submit per-axis readout job"
  if [ -f benchmarking/eval_scripts/scripts/readout_axes.py ]; then
    JOB=$(sbatch --parsable benchmarking/eval_scripts/scripts/slurm_readout_axes.sh) && echo "submitted job $JOB"
    echo "${JOB:-}" > lab_runs/status/readout_axes_jobid
  else
    echo "readout_axes.py missing: pull did not bring it in"
  fi

  echo; echo "== 4 CT-RATE data for task 7"
  find $U "$HOME/.cache/huggingface" -maxdepth 7 \( -name "train_reports.csv" -o -name "validation_reports.csv" -o -name "*metadata*.csv" \) 2>/dev/null
  echo "-- directories named train / train_fixed / valid_fixed (with .nii.gz counts):"
  find $U "$HOME/.cache/huggingface" -maxdepth 7 -type d \( -name train -o -name train_fixed -o -name valid_fixed \) 2>/dev/null | while read -r d; do
    echo "$d  $(find "$d" -name '*.nii.gz' 2>/dev/null | wc -l) volumes  $(du -sh "$d" 2>/dev/null | cut -f1)"
  done
  echo "-- HF cache entries:"; ls "$HOME/.cache/huggingface/hub" 2>/dev/null | grep -i -E "ct-rate|ct_rate|reason"
  echo "-- disk:"; df -h $U "$HOME" 2>/dev/null
  quota -s 2>/dev/null || true

  echo; echo "== 5 NV-Reason-CT code files (no weights)"
  S=$(ls -d "$HOME"/.cache/huggingface/hub/models--nvidia--NV-Reason-CT/snapshots/* 2>/dev/null | head -1)
  echo "snapshot: ${S:-not found}"
  if [ -n "${S:-}" ]; then
    for f in "$S"/*.py "$S"/*.json "$S"/*.jinja "$S"/requirements.txt; do
      [ -e "$f" ] && [ "$(basename "$f")" != "model.safetensors.index.json" ] && cp -L "$f" lab_runs/nvreason_model_code/
    done
    if [ -d "$S/image_processor_3d" ]; then
      mkdir -p lab_runs/nvreason_model_code/image_processor_3d
      find -L "$S/image_processor_3d" -maxdepth 2 -type f \( -name "*.py" -o -name "*.json" \) -size -2M \
        -exec cp -L {} lab_runs/nvreason_model_code/image_processor_3d/ \;
    fi
    ls -la lab_runs/nvreason_model_code lab_runs/nvreason_model_code/image_processor_3d 2>/dev/null
  fi

  echo; echo "== 6 training libraries in the nvreason env"
  /home/mashrafi.monon/.conda/envs/nvreason/bin/python -c "
import importlib
for m in ['torch', 'transformers', 'peft', 'accelerate', 'nibabel', 'monai']:
    try:
        print(m, importlib.import_module(m).__version__)
    except Exception as e:
        print(m, 'MISSING', type(e).__name__)
" 2>&1 | grep -v -i warning
  squeue -u "$(whoami)"
}

main > "$OUT" 2>&1
cat "$OUT"

echo; echo "== pushing results"
git add lab_runs
git commit -q -m "Lab status $D: readout-axes job, CT-RATE check, NV-Reason-CT code files" && echo committed
REMOTE=$(git config branch.finetune.remote || echo personal)
timeout 180 git push "$REMOTE" HEAD:finetune && echo "PUSHED to $REMOTE finetune" || echo "PUSH FAILED (retry: git push $REMOTE HEAD:finetune)"
