exp1.slurm#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH -p classc -A sxg125_csds438 -N 1 -n 4 --mem=16gb
#SBATCH --job-name="contact-tracing:exp2"
#SBATCH --mail-user=rdt46@case.edu
#SBATCH --mail-type=ALL

EXPERIMENT=2
PYTHON_VERSION="3.8.6"

module swap intel gcc
module load python/$PYTHON_VERSION

export PYTHONUSERBASE=$HOME/.usr/local/python/$PYTHON_VERSION
mkdir -p $PYTHONUSERBASE

cp ~/contact-tracing $PFSDIR
cd $PFSDIR

pip install --upgrade pip
pip install -r ~/contact-tracing/requirements.txt --user
pip install -e ~/contact-tracing

cd ~/contact-tracing/evaluation

python3 eval.py $EXPERIMENT
