#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH -p batch -A exa208 -N 1 -n 4 --mem=8gb
#SBATCH --job-name="r-workplace"
#SBATCH --mail-user=rdt46@case.edu
#SBATCH --mail-type=ALL

PYTHON_VERSION="3.8.6"
PACKAGE="contact-tracing"
TYPE="real"
SETTING="workplace"

module swap intel gcc
module load python/$PYTHON_VERSION

cp $HOME/$PACKAGE $PFSDIR
cd $PFSDIR
cd $HOME/$PACKAGE/evaluation

python3 evaluate.py $TYPE \
    --setting $SETTING --path data/$SETTING.txt --seed 12345
