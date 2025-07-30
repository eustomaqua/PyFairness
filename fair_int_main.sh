#!/bin/bash
# The partition is the queue you want to run on. standard is gpu and can be ommitted.
# number of independent tasks we are going to start in this script
# number of cpus we want to allocate for each program
# We expect that our program should not run longer than 2 days
# Note that a program will be killed once it exceeds this time!
# Skipping many options! see man sbatch


#SBATCH -p gpu --gres=gpu:0
#SBATCH --job-name=icml25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00


# From here on, we can start our program

# chmod +x fair_int_main.sh
# cd ~/Singdocker
# module load singularity
# singularity run enfair.sif
# source activate ensem
# cd ~/GitH/PyFairness


PREP=min_max

EXPT=KF_exp1b
for DATA in ricci german ppr ppvr # adult
do
  python fair_int_exec.py -exp $EXPT -dat $DATA -pre $PREP
done

EXPT=KF_exp1c
for DATA in ricci german ppr ppvr # adult
do
  python fair_int_exec.py -exp $EXPT -dat $DATA -pre $PREP
done

# conda deactivate
# exit
