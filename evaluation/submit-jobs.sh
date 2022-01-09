#!/bin/sh

bash scripts/setup.sh
sbatch scripts/real-conference.slurm
sbatch scripts/real-workplace.slurm
sbatch scripts/real-highschool12.slurm
sbatch scripts/parameters-lfr.slurm
sbatch scripts/parameters-power.slurm
sbatch scripts/parameters-geometric.slurm
sbatch scripts/scalability-lfr.slurm
sbatch scripts/scalability-power.slurm
sbatch scripts/scalability-geometric.slurm