#!/bin/bash

cd /p/gscratchr/karande1/
. spack/share/spack/setup-env.sh

spack activate py-ipython
spack activate py-keras
spack activate py-tqdm
spack activate py-scikit-learn
spack activate py-requests

module load py-ipython-5.1.0-gcc-4.8.5-l5wsryf
module load python-2.7.13-gcc-4.8.5-mrskb7k

cd Benchmarks/Pilot2/P2B1/
DATE=`date +%m%d%Y_%H%M%S`
save_path="/p/gscratchr/karande1/Benchmarks/Pilot2/P2B1/molecular_AE_results/$DATE"
mkdir $save_path
mpirun -np 1 ipython -i p2b1_baseline_keras2_mol_only.py -- --save-dir $save_path --case Full --config-file p2b1_default_model.txt

echo 'Done'
