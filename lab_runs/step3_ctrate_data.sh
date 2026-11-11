#!/bin/bash
# Run on the lab (a node with internet, e.g. ws-l4-021), from the repo root:  bash lab_runs/step3_ctrate_data.sh
# Task 7 data: CT-RATE train reports -> training list -> background download of the selected train volumes.
# 1 quota check   2 download train_reports.csv   3 build the training list (pushed)   4 install peft (no deps)
# 5 start the volume download in the background (nohup, survives logout; resumable, rerun-safe)
set -u
cd "$(dirname "$0")/.."
D=$(date +%Y-%m-%d_%H%M)
OUT=lab_runs/status/${D}_step3.txt
DV=/l/users/mashrafi.monon/MICCAI2026-3DMedVLMS/3D_VLM_Spatial/dataset/data_volumes   # mirrors the HF repo layout
PY=/home/mashrafi.monon/.conda/envs/nvreason/bin/python
HF=$(command -v hf || ls /home/mashrafi.monon/.conda/envs/nvreason/bin/hf 2>/dev/null || command -v huggingface-cli)
mkdir -p lab_runs/status finetuning/nvreason_spatial/data

main() {
  echo "host $(hostname)  date $(date)  hf cli: ${HF:-none}"
  echo; echo "== 1 quota on /l (stop if under ~800 GB left)"
  lfs quota -h -u "$(whoami)" /l 2>/dev/null || echo "lfs quota not available"
  $HF auth whoami 2>/dev/null || $HF whoami 2>/dev/null || echo "not logged in to HF (CT-RATE is gated: run 'hf auth login')"

  echo; echo "== 2 train reports"
  R=$DV/dataset/radiology_text_reports/train_reports.csv
  [ -s "$R" ] || $HF download ibrahimhamamci/CT-RATE dataset/radiology_text_reports/train_reports.csv \
      --repo-type dataset --local-dir "$DV" 2>&1 | tail -2
  ls -la "$R" || { echo "train_reports.csv download FAILED"; return 1; }

  echo; echo "== 3 training list (4,000 scans, seed 0)"
  $PY finetuning/nvreason_spatial/build_spatial_train_set.py --reports "$R" \
      --out-dir finetuning/nvreason_spatial/data --max-volumes 4000 || return 1

  echo; echo "== 4 peft into the nvreason env (no deps, so torch/CUDA wheels are not touched)"
  $PY -m pip install -q --no-deps peft 2>&1 | tail -2
  $PY -c "import peft; print('peft', peft.__version__)"

  echo; echo "== 5 volume download in the background"
  if pgrep -u "$(whoami)" -f download_ctrate_volumes.sh >/dev/null; then
    echo "already running"
  else
    nohup bash lab_runs/download_ctrate_volumes.sh > lab_runs/status/download.log 2>&1 &
    echo "started, pid $!  (progress: bash lab_runs/step4_download_status.sh)"
  fi
}

main > "$OUT" 2>&1
cat "$OUT"

echo; echo "== pushing"
git add lab_runs/status/*_step3.txt finetuning/nvreason_spatial/data
git commit -q -m "Lab status $D: CT-RATE train reports, training list, download started" && echo committed
REMOTE=$(git config branch.finetune.remote || echo personal)
timeout 180 git push "$REMOTE" HEAD:finetune && echo "PUSHED to $REMOTE finetune" || echo "PUSH FAILED (retry: git push $REMOTE HEAD:finetune)"
