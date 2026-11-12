#!/bin/bash
# Run on the lab from the repo root:  nohup bash lab_runs/side_lean.sh > /dev/null 2>&1 &
# Submits the side-lean analysis as a SLURM CPU job, waits for it, copies the results to lab_runs/results/side_lean/
# and pushes them.
set -u
cd "$(dirname "$0")/.."
mkdir -p lab_runs/results/side_lean
A=benchmarking/reports/analysis
sbatch --wait -p ws-ia -c 8 --mem=48G -t 02:00:00 -J side-lean -o logs/side_lean_%j.out -e logs/side_lean_%j.err \
  --wrap "export OMP_NUM_THREADS=8; ~/venvs/analysis/bin/python benchmarking/eval_scripts/scripts/readout_side_lean.py --stem $A/readout_all --out-dir $A/readout_side_lean"
cp $A/readout_side_lean/* lab_runs/results/side_lean/ 2>/dev/null
cp "$(ls -t logs/side_lean_*.err | head -1)" lab_runs/results/side_lean/side_lean.err 2>/dev/null
git add lab_runs/results/side_lean
git commit -q -m "Side-lean analysis results $(date +%Y-%m-%d_%H%M)"
REMOTE=$(git config branch.finetune.remote || echo personal)
timeout 180 git push "$REMOTE" HEAD:finetune
