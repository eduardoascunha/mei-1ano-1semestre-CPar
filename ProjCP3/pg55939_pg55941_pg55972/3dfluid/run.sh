#!/bin/bash

#SBATCH --partition=cpar
#SBATCH --time=0-00:2:00
#SBATCH --constraint=k20
#SBATCH --exclusive
#SBATCH -W

echo "-------------------------------------------------Run-------------------------------------------------"
time nvprof ./fluid_sim
echo "-------------------------------------------------End-------------------------------------------------"
