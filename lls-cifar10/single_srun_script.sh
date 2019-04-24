#!/bin/bash

#SBATCH --partition mllab

#SBATCH --nodes 1

#SBATCH --ntasks 1

#SBATCH --cpus-per-task 10

#SBATCH --gres gpu:4

#SBATCH --time 50:00:00

#SBATCH --mem 120000

#SBATCH --output slurm-%j.out

#SBATCH --qos normal

hostname

srun python -u $1 $2
