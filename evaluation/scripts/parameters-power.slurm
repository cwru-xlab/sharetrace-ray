#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH -p batch -A exa208 -N 1 -n 4 --mem=8gb
#SBATCH --job-name="parameters-power"
#SBATCH --mail-user=rdt46@case.edu
#SBATCH --mail-type=ALL

PYTHON_VERSION="3.8.6"
PACKAGE="contact-tracing"

module swap intel gcc
module load python/$PYTHON_VERSION

export PYTHONUSERBASE=$HOME/.usr/local/python/$PYTHON_VERSION
mkdir -p $PYTHONUSERBASE

cp ~/$PACKAGE $PFSDIR
cd $PFSDIR

pip install --upgrade pip
pip install -r ~/$PACKAGE/requirements.txt --user
pip install -e ~/$PACKAGE

cd ~/$PACKAGE/evaluation

python3 evaluate.py parameters --graph power --seed 12345

cp -ru * $SLURM_SUBMIT_DIR
rm -rf "$PFSDIR"/*
