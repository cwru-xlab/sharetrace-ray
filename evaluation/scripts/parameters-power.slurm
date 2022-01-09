#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH -p batch -A exa208 -N 1 -n 4 --mem=16gb
#SBATCH --job-name="p-power"
#SBATCH --mail-user=rdt46@case.edu
#SBATCH --mail-type=ALL

PYTHON_VERSION="3.8.6"
PACKAGE="contact-tracing"
TYPE="parameters"
GRAPH="power"

module load python/$PYTHON_VERSION

cp $HOME/$PACKAGE $PFSDIR
cd $PFSDIR
cd $HOME/$PACKAGE/evaluation

python3 evaluate.py $TYPE --graph $GRAPH --seed 12345
