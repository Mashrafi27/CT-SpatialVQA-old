#!/bin/bash
# Run on the lab after step 1, from the repo root:  bash lab_runs/step2_collect_readout_axes.sh
# If the per-axis readout job is finished, copies its summary, plots and log into lab_runs/results/ and pushes them.
set -u
cd "$(dirname "$0")/.."
JOB=$(cat lab_runs/status/readout_axes_jobid 2>/dev/null)
[ -z "$JOB" ] && { echo "no job id: run step 1 first"; exit 1; }
STATE=$(squeue -h -j "$JOB" -o %T 2>/dev/null)
if [ -n "$STATE" ]; then
  echo "job $JOB is still $STATE; run this again later"
  tail -3 "logs/readout_axes_${JOB}.out" 2>/dev/null
  exit 0
fi

R=lab_runs/results/readout_axes
mkdir -p $R
cp benchmarking/reports/analysis/readout_axes/* $R/ 2>/dev/null
cp "logs/readout_axes_${JOB}.out" "logs/readout_axes_${JOB}.err" $R/ 2>/dev/null
if grep -q AXES_DONE "$R/readout_axes_${JOB}.out" 2>/dev/null; then echo "job $JOB finished OK"; else echo "job $JOB did NOT finish OK; log copied for diagnosis"; tail -20 "$R/readout_axes_${JOB}.err" 2>/dev/null; fi
ls -la $R

git add lab_runs
git commit -q -m "Readout per-axis results (job $JOB)" && echo committed
REMOTE=$(git config branch.finetune.remote || echo personal)
timeout 180 git push "$REMOTE" HEAD:finetune && echo "PUSHED to $REMOTE finetune" || echo "PUSH FAILED (retry: git push $REMOTE HEAD:finetune)"
