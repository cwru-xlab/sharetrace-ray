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

python3 -m pip install --upgrade pip
python3 -m pip install -r $HOME/$PACKAGE/requirements.txt --user
python3 -m pip install -e $HOME/$PACKAGE

cp -r $HOME/$PACKAGE $PFSDIR
cd $PFSDIR/$PACKAGE/evaluation

python3 evaluate.py $TYPE \
    --setting $SETTING --path data/$SETTING.txt --seed 12345
