#!/bin/sh

PACKAGE="contact-tracing"
PYTHON_VERSION="3.8.6"

module swap intel gcc
module load python/$PYTHON_VERSION

export PYTHONUSERBASE=$HOME/.usr/local/python/$PYTHON_VERSION
mkdir -p $PYTHONUSERBASE
pip install --upgrade pip
pip install -r $HOME/$PACKAGE/requirements.txt --user
pip install -e $HOME/$PACKAGE