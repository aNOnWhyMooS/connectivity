<<com
Installs packages necessary for hf script's finetuning.
com

#!/bin/bash
set -e
conda install transformers -y
conda install cuda-nvcc cudatoolkit=11.8 -c nvidia -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
CONDA_OVERRIDE_CUDA="11.8" conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia -y
conda install -c conda-forge flax -y
conda install -c conda-forge git-lfs -y
pip install -U accelerate datasets tensorboard tensorflow
pip install git+https://github.com/Jeevesh8/trfrmr_weight_matching/#subdirectory=src
pip install git+https://github.com/anonwhymoos/connectivity@cur_dev#subdirectory=src
pip install GitPython sentencepiece tabulate rouge_score scikit-image scikit-learn