#!/bin/bash
# Run on the lab from the repo root:  bash lab_runs/step4_download_status.sh
# Reports download progress (and restarts the download if it died before finishing), then pushes the status.
set -u
cd "$(dirname "$0")/.."
DV=/l/users/mashrafi.monon/MICCAI2026-3DMedVLMS/3D_VLM_Spatial/dataset/data_volumes
LIST=finetuning/nvreason_spatial/data/volumes.txt
OUT=lab_runs/status/download_status.txt
{
  echo "$(date)  host $(hostname)"
  have=0
  while read -r v; do
    s=${v%_*}; s=${s%.nii.gz}; p=${s%_*}
    [ -s "$DV/dataset/train_fixed/$p/$s/$v" ] && have=$((have + 1))
  done < "$LIST"
  echo "volumes on disk: $have of $(wc -l < "$LIST")"
  du -sh "$DV/dataset/train_fixed" 2>/dev/null
  tail -5 lab_runs/status/download.log 2>/dev/null
  if pgrep -u "$(whoami)" -f download_ctrate_volumes.sh >/dev/null; then
    echo "download running"
  elif ! grep -q DOWNLOAD_FINISHED lab_runs/status/download.log 2>/dev/null; then
    nohup bash lab_runs/download_ctrate_volumes.sh >> lab_runs/status/download.log 2>&1 &
    echo "download was not running and not finished: restarted (pid $!)"
  fi
} > "$OUT" 2>&1
cat "$OUT"
git add "$OUT"
git commit -q -m "Download status $(date +%Y-%m-%d_%H%M)" && echo committed
REMOTE=$(git config branch.finetune.remote || echo personal)
timeout 180 git push "$REMOTE" HEAD:finetune && echo "PUSHED" || echo "PUSH FAILED (retry: git push $REMOTE HEAD:finetune)"
