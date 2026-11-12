#!/bin/bash
# Downloads the CT-RATE train volumes listed in finetuning/nvreason_spatial/data/volumes.txt, in batches of 100, skipping
# files already on disk or already cropped into the cache. Runs on the login node / a workstation (compute nodes have no
# internet) under nohup; safe to rerun (hf download resumes partial files).
# Every file it fetches is appended to lab_runs/status/downloaded_by_us.txt: only those may be deleted by the cache job.
# Waits while more than MAX_PENDING raw files (~150 MB each) are waiting to be cropped, so disk use stays bounded.
# train_123_a_1.nii.gz -> dataset/train_fixed/train_123/train_123_a/train_123_a_1.nii.gz
set -u
cd "$(dirname "$0")/.."
DV=/l/users/mashrafi.monon/MICCAI2026-3DMedVLMS/3D_VLM_Spatial/dataset/data_volumes
CACHE=/l/users/mashrafi.monon/nvreason_spatial_cache
HF=$(command -v hf || ls /home/mashrafi.monon/.conda/envs/nvreason/bin/hf 2>/dev/null || command -v huggingface-cli)
LIST=finetuning/nvreason_spatial/data/volumes.txt
OWNED=lab_runs/status/downloaded_by_us.txt
MAX_PENDING=${MAX_PENDING:-300}
touch "$OWNED"

todo=()
while read -r v; do
  [ -z "$v" ] && continue
  s=${v%_*}; p=${s%_*}                                 # train_123_a_1.nii.gz -> s=train_123_a, p=train_123
  f="dataset/train_fixed/$p/$s/$v"
  [ -s "$DV/$f" ] || [ -s "$CACHE/${v%.nii.gz}.npy" ] || todo+=("$f")
done < "$LIST"
echo "$(date) ${#todo[@]} of $(wc -l < "$LIST") volumes to download"

pending() { local n=0; while read -r f; do [ -e "$f" ] && n=$((n + 1)); done < "$OWNED"; echo $n; }

for ((i = 0; i < ${#todo[@]}; i += 100)); do
  while [ "$(pending)" -gt "$MAX_PENDING" ]; do sleep 120; done
  batch=("${todo[@]:i:100}")
  for try in 1 2 3; do
    $HF download ibrahimhamamci/CT-RATE "${batch[@]}" --repo-type dataset --local-dir "$DV" --max-workers 8 \
      > /dev/null 2>&1 && break
    echo "$(date) batch $i try $try failed"; sleep 60
  done
  for f in "${batch[@]}"; do [ -s "$DV/$f" ] && echo "$DV/$f" >> "$OWNED"; done
  echo "$(date) done through $((i + 100)), pending crop $(pending)"
done
echo "$(date) DOWNLOAD_FINISHED"
