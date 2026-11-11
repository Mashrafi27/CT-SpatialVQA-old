#!/bin/bash
# Run on the lab from the repo root:  nohup bash lab_runs/disk_usage.sh > /dev/null 2>&1 &
# Read-only: where the ~2 TB under /l/users/mashrafi.monon goes, and where CT-RATE scans / archives already are.
# Takes a while on Lustre (minutes to ~1 h); commits lab_runs/status/disk_usage.txt and pushes when done.
set -u
cd "$(dirname "$0")/.."
U=/l/users/mashrafi.monon
OUT=lab_runs/status/disk_usage.txt
{
  echo "$(date)  host $(hostname)"
  lfs quota -h -u "$(whoami)" /l 2>/dev/null
  echo; echo "== biggest folders (2 levels down, >= 10 GB)"
  du -h --max-depth=2 "$U" 2>/dev/null | sort -rh | awk '$1 ~ /T$/ || ($1 ~ /G$/ && $1+0 >= 10)' | head -60
  echo; echo "== CT-RATE scans anywhere (train_*/valid_* .nii.gz), grouped by folder 3 levels up"
  find "$U" -type f \( -name "train_*.nii.gz" -o -name "valid_*.nii.gz" \) 2>/dev/null \
    | awk -F/ '{n=split($0,a,"/"); k=""; for(i=1;i<=n-3;i++) k=k a[i] "/"; c[k]++} END {for (k in c) print c[k], k}' | sort -rn | head -30
  echo; echo "== archives >= 1 GB (tar / zip / npz / pt)"
  find "$U" -type f \( -name "*.tar*" -o -name "*.zip" -o -name "*.npz" -o -name "*.pt" -o -name "*.safetensors" \) -size +1G \
    -printf "%s\t%p\n" 2>/dev/null | sort -rn | awk -F'\t' '{printf "%7.1f GB  %s\n", $1/1e9, $2}' | head -60
  echo; echo "== HF dataset cache (in /home, separate quota)"
  du -sh "$HOME"/.cache/huggingface/hub/datasets--* 2>/dev/null
  echo; echo "$(date) DONE"
} > "$OUT" 2>&1
git add "$OUT"
git commit -q -m "Disk usage report $(date +%Y-%m-%d_%H%M)"
REMOTE=$(git config branch.finetune.remote || echo personal)
timeout 180 git push "$REMOTE" HEAD:finetune
