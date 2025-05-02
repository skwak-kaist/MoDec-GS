#!/bin/bash
# s.kwak@etri.re.kr

set -e

env_name=modecgs
echo "Installing MoDec-GS in conda environment:" $env_name

bash setup_modec_env.sh $env_name

# activate vcm_decoder environment
eval "$(conda shell.bash hook)"
conda activate $env_name

echo "Checking Python version"
python --version

echo "Installing requirements"
pip install -r requirements.txt

echo "Installing submodules"

pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn

echo "Install is complete"

