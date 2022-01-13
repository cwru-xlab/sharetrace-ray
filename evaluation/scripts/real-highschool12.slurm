#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH -p batch -A exa208 -N 1 -n 4 --mem=8gb
#SBATCH --job-name="r-highschool12"
#SBATCH --mail-user=rdt46@case.edu
#SBATCH --mail-type=ALL

PYTHON_VERSION="3.8.6"
METIS_VERSION="5.1.0"
CMAKE_VERSION="3.19.3"
METIS_PREFIX=$PFSDIR/metis
METIS_PATH=sharetrace/metis-$METIS_VERSION
PACKAGE="contact-tracing"
TYPE="real"
SETTING="highschool12"

module swap intel gcc
module load python/$PYTHON_VERSION
module load cmake/$CMAKE_VERSION

cp -r $HOME/$PACKAGE $PFSDIR
cd $PFSDIR/$PACKAGE

# Ref: https://metis.readthedocs.io/en/latest/
(cd $METIS_PATH && make distclean && make config shared=1 prefix=$METIS_PREFIX &&
make install)

# Ref: https://superuser.com/a/595165
export METIS_DLL=$METIS_PREFIX/lib/libmetis.so

cd evaluation

python3 evaluate.py $TYPE \
    --setting $SETTING --path data/$SETTING.csv --seed 12345

cp -r logs/$TYPE/$SETTING* $HOME/$PACKAGE/evaluation/logs/$TYPE
