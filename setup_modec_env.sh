# !/bin/bash 
# s.kwak@etri.re.kr

# set up conda environment for MoDec-GS
conda create --name $1 python=3.7.16 -y

eval "$(conda shell.bash hook)"
conda activate $1

pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116


