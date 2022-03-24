#!/bin/sh

bash scripts/setup.sh
for i in {1..5} ; do
    sbatch scalability-geometric-$i.slurm
    sbatch scalability-lfr-$i.slurm
    sbatch scalability-power-$i.slurm
done
