#!/bin/bash
#PBS -N create_venv
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=4:mem=16GB:scratch_local=10gb
#PBS -j oe
#PBS -o create_venv.out

module purge
module add python/3.10.4-gcc-8.3.0-ovkjwzd

# Redirect temporary files to scratch to avoid /tmp quota
mkdir -p $SCRATCHDIR/tmp
export TMPDIR=$SCRATCHDIR/tmp
mkdir -p $SCRATCHDIR/pip_cache
export PIP_CACHE_DIR=$SCRATCHDIR/pip_cache

# Create a directory for the venv in storage
VENV_DIR=/storage/brno2/home/$USER/KNN/mer_env
mkdir -p $VENV_DIR

python3 -m venv $VENV_DIR

# Activate the venv
source $VENV_DIR/bin/activate

pip install --upgrade pip

pip install torch transformers datasets editdistance pillow plasTeX numpy albumentations accelerate>=1.1.0

pip cache purge

echo "Virtual environment created in $VENV_DIR"
echo "To use it later: source $VENV_DIR/bin/activate"
rm -rf $SCRATCHDIR/*