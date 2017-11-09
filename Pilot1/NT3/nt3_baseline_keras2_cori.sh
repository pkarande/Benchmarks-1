#!/bin/bash

function_to_fork() {
  top -b -n 240 -d 60 -u brettin > $SLURM_JOBID.$$.top
}
function_to_fork &

python nt3_baseline_keras2.py
