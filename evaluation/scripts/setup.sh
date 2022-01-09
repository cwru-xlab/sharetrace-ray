#!/bin/sh

PACKAGE="contact-tracing"
PYTHON_VERSION="3.8.6"

module swap intel gcc
module load python/$PYTHON_VERSION

export PYTHONUSERBASE=$HOME/.usr/local/python/$PYTHON_VERSION
mkdir -p $PYTHONUSERBASE
python3 -m pip install --upgrade pip
python3 -m pip install -r $HOME/$PACKAGE/requirements.txt --user
python3 -m pip install -e $HOME/$PACKAGE
