#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH -p classc -A sxg125_csds438 -N 4 --mem=16gb

EXPERIMENT = 1
PYTHON_VERSION = "3.8.6"

module swap intel gcc
module load python/$PYTHON_VERSION

export PYTHONUSERBASE=$HOME/.usr/local/python/$PYTHON_VERSION
mkdir -p $PYTHONUSERBASE

cp ~/contact-tracing $PFSDIR
cd $PFSDIR

pip install -r ~/contact-tracing/requirements.txt --user
python3 -m pip install ~/contact-tracing

cd ~/contact-tracing/evaluation

python3 eval.py $EXPERIMENT