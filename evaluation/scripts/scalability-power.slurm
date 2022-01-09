#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH -p batch -A exa208 -N 1 -n 8 --mem=64gb
#SBATCH --job-name="s-power"
#SBATCH --mail-user=rdt46@case.edu
#SBATCH --mail-type=ALL

PYTHON_VERSION="3.8.6"
PACKAGE="contact-tracing"
TYPE="scalability"
GRAPH="power"

cp $HOME/$PACKAGE $PFSDIR
cd $PFSDIR
cd $HOME/$PACKAGE/evaluation

python3 evaluate.py $TYPE --graph $GRAPH --seed 12345
