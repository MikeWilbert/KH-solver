#!/bin/bash -x
#SBATCH --account=specturb
#SBATCH --nodes=8
#SBATCH --tasks-per-node=48
#SBATCH --output=log/mpi_%j.out
#SBATCH --error=log/mpi_%j.err
#SBATCH --time=00:01:00
#SBATCH --partition=batch

srun --overlap ./mike_phy
