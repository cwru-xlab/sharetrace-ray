#!/bin/sh

PACKAGE="contact-tracing"

export PYTHONUSERBASE=$HOME/.usr/local/python/$PYTHON_VERSION
mkdir -p $PYTHONUSERBASE
pip install --upgrade pip
pip install -r ~/$PACKAGE/requirements.txt --user
pip install -e ~/$PACKAGE