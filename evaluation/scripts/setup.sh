#!/bin/sh

PYTHON_VERSION="3.8.6"
METIS_VERSION="5.1.0"
CMAKE_VERSION="3.19.3"
PACKAGE="contact-tracing"

module swap intel gcc
module load python/$PYTHON_VERSION
module load cmake/$CMAKE_VERSION

export PYTHONUSERBASE=$HOME/.usr/local/python/$PYTHON_VERSION
mkdir -p $PYTHONUSERBASE

# Ref: https://metis.readthedocs.io/en/latest/
(cd $HOME/$PACKAGE/sharetrace/metis-$METIS_VERSION && make config shared=1)

python3 -m pip install --upgrade pip
python3 -m pip install -r $HOME/$PACKAGE/requirements.txt --user
python3 -m pip install -e $HOME/$PACKAGE
