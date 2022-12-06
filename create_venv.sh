#!/bin/bash
# this script is intended to be run where it is found

export DATA_DIR=/home/npetroce/data/
export PYTHONPATH=$PYTHONPATH:$PWD
python3.9 -m pip install virtualenv
python3.9 -m virtualenv models_env
source models_env/bin/activate
pip install -U pip
pip install --default-timeout=1000 -r  official/requirements.txt
pip install tensorflow
pip install tensorflow-text-nightly
pip install tensorflow-text
pip install datasets

echo environment ready, activate with "source models_env/bin/activate" next time you log in