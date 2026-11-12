#!/bin/bash
#SBATCH --job-name=nvr-cache
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/nvr_cache_%j.out
#SBATCH --error=logs/nvr_cache_%j.err
# CPU job: crops downloaded CT-RATE train volumes with NV-Reason-CT's own preprocessing into a float16 cache and deletes
# each raw file our download fetched once its crop is saved. Polls for new downloads; resubmit to resume.
cd /l/users/mashrafi.monon/MICCAI2026-3DMedVLMS/CT-SpatialVQA
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DV=/l/users/mashrafi.monon/MICCAI2026-3DMedVLMS/3D_VLM_Spatial/dataset/data_volumes
/home/mashrafi.monon/.conda/envs/nvreason/bin/python finetuning/nvreason_spatial/cache_volumes.py \
  --volumes finetuning/nvreason_spatial/data/volumes.txt --data-root $DV \
  --cache-dir /l/users/mashrafi.monon/nvreason_spatial_cache --owned lab_runs/status/downloaded_by_us.txt \
  --delete-raw --wait
