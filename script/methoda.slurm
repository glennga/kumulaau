#!/bin/bash

## Our SLURM options:
#SBATCH --job-name=modela-mcmc
#SBATCH --partition=community.q
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=10000
#SBATCH --error=/home/glennga/lus/methoda-%A_%a.err
#SBATCH --output=/home/glennga/lus/methoda-%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=glennga@hawaii.edu

## Grab our environment variables and load our modules.
source ~/.bash_profile
source ~/.bashrc

## Not sure what this is for... but it's in the example!
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

conda activate /home/glennga/apps/software/coaenv
/home/glennga/apps/micro-coa/script/methoda.sh ${SLURM_ARRAY_TASK_ID}
