#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH -p classc -A sxg125_csds438 -N 1 -n 8 --mem=16gb
#SBATCH --job-name="real-highschool11"
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

python3 eval.py real-world \
    --setting highschool11 --path data/highschool_2011.csv --seed 12345

cp -ru * $SLURM_SUBMIT_DIR
rm -rf "$PFSDIR"/*
