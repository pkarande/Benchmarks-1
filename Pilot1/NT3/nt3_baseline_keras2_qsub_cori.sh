#!/bin/bash


#SBATCH -N 2 
#SBATCH -p debug 
#SBATCH -C knl,quad,cache
#SBATCH -t 30:00
#SBATCH -L SCRATCH



set -x


module load python/2.7-anaconda
module load darshan
export LD_LIBRARY_PATH=/usr/common/software/darshan/3.1.4/lib:$LD_LIBRARY_PATH

CONDA_ENV=horovod-tf2
source activate $CONDA_ENV

export OMP_NUM_THREADS=136
export KMP_BLOCKTIME=30
export KMP_SETTINGS=1
export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR

srun --export=OMP_NUM_THREADS,KMP_BLOCKTIME,KMP_SETTINGS,KMP_AFFINITY $DIR/nt3_baseline_keras2_cori.sh

source deactivate
