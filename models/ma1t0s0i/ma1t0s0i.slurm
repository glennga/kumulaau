#!/bin/bash

## Our SLURM options:
#SBATCH --job-name=ma1t0s0i
#SBATCH --partition=community.q
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=10000
#SBATCH --error=/home/glennga/lus/out/ma1t0s0i-%A_%a.err
#SBATCH --output=/home/glennga/lus/out/ma1t0s0i	-%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=glennga@hawaii.edu

## Grab our environment variables and load our modules.
source ~/.bash_profile
source ~/.bashrc
module load tools/GSL/2.3-intel-2017a
module unload binutils/2.27-GCCcore-6.3.0
module unload GCCcore/6.3.0
module load GCC/7.1.0-2.28

## Not sure what this is for... but it's in the example!
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

conda activate /home/glennga/apps/software/coaenv
/home/glennga/apps/kumulaau/script/ma1t0s0i/ma1t0s0i.sh /home/glennga/lus/ma1t0s0i-${SLURM_ARRAY_TASK_ID}.db"
