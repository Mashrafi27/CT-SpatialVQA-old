#!/bin/bash
#SBATCH --job-name=readout-axes
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/readout_axes_%j.out
#SBATCH --error=logs/readout_axes_%j.err
# CPU job: logit lens for the axis words, then the per-axis readout analysis (lr / si / ap / ml).
# Never run these on the login node (multi-GB arrays + torch matmuls).
cd /l/users/mashrafi.monon/MICCAI2026-3DMedVLMS/CT-SpatialVQA
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
A=benchmarking/reports/analysis
E=benchmarking/reports/llm_eval
[ -f $A/readout_all_axis_lens.npz ] || /home/mashrafi.monon/.conda/envs/nvreason/bin/python \
  benchmarking/eval_scripts/scripts/readout_axes.py lens --stem $A/readout_all --out $A/readout_all_axis_lens.npz || exit 1
~/venvs/analysis/bin/python benchmarking/eval_scripts/scripts/readout_axes.py analyse --stem $A/readout_all \
  --lens $A/readout_all_axis_lens.npz --out-dir $A/readout_axes \
  --judge-image $E/nv_reason_ct_qwen_eval.json --judge-blind $E/nv_reason_ct_blind_qwen_eval.json && echo AXES_DONE
