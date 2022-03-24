#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH -p batch -A exa208 -N 1 -n 8 --mem=16gb
#SBATCH --job-name="s-lfr-1"
#SBATCH --mail-user=rdt46@case.edu
#SBATCH --mail-type=ALL

PYTHON_VERSION="3.8.6"
PACKAGE="contact-tracing"
TYPE="scalability"
GRAPH="lfr"

module swap intel gcc
module load python/$PYTHON_VERSION

cp -r $HOME/$PACKAGE $PFSDIR
cd $PFSDIR/$PACKAGE/evaluation

python3 evaluate.py $TYPE \
    --graph $GRAPH --start 100 --stop 2100 --step 100 --seed 12345

cp logs/$TYPE/$GRAPH* $HOME/$PACKAGE/evaluation/logs/$TYPE
