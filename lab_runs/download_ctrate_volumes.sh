#!/bin/bash
# Downloads the CT-RATE train volumes listed in finetuning/nvreason_spatial/data/volumes.txt, in batches of 100, skipping
# files already on disk. Started by step3 under nohup; safe to rerun (hf download resumes partial files).
# train_123_a_1.nii.gz -> dataset/train_fixed/train_123/train_123_a/train_123_a_1.nii.gz
set -u
cd "$(dirname "$0")/.."
DV=/l/users/mashrafi.monon/MICCAI2026-3DMedVLMS/3D_VLM_Spatial/dataset/data_volumes
HF=$(command -v hf || ls /home/mashrafi.monon/.conda/envs/nvreason/bin/hf 2>/dev/null || command -v huggingface-cli)
LIST=finetuning/nvreason_spatial/data/volumes.txt

todo=()
while read -r v; do
  [ -z "$v" ] && continue
  s=${v%_*}; s=${s%.nii.gz}; p=${s%_*}               # train_123_a_1.nii.gz -> s=train_123_a, p=train_123
  f="dataset/train_fixed/$p/$s/$v"
  [ -s "$DV/$f" ] || todo+=("$f")
done < "$LIST"
echo "$(date) ${#todo[@]} of $(wc -l < "$LIST") volumes to download"

for ((i = 0; i < ${#todo[@]}; i += 100)); do
  for try in 1 2 3; do
    $HF download ibrahimhamamci/CT-RATE "${todo[@]:i:100}" --repo-type dataset --local-dir "$DV" --max-workers 8 \
      > /dev/null 2>&1 && break
    echo "$(date) batch $i try $try failed"; sleep 60
  done
  echo "$(date) done through $((i + 100))"
done
echo "$(date) DOWNLOAD_FINISHED"
