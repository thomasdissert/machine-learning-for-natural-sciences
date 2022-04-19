#!/bin/bash
set -e

name="aimat"

conda update -y -n base -c defaults conda

conda env create -f environment.yml

# this way, activating conda from an uninitialized shell is avoided
conda run -n $name jupyter nbextension install --sys-prefix --py nbgrader --overwrite
conda run -n $name jupyter nbextension enable --sys-prefix --py nbgrader
conda run -n $name jupyter serverextension enable --sys-prefix --py nbgrader

echo "====================== DONE ====================="
echo "Please run: 'conda activate $name' to activate your conda environment."
