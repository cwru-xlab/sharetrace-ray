#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH -p classc -A sxg125_csds438 -N 1 -n 8 --mem=32gb
#SBATCH --job-name="scalability-lfr"
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

python3 eval.py scalability --graph lfr --seed 12345

cp -ru * $SLURM_SUBMIT_DIR
rm -rf "$PFSDIR"/*
