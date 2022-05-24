#!/bin/bash
mkdir ./ext3
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b -p ./ext3/miniconda3
echo "source ./ext3/miniconda3/etc/profile.d/conda.sh
export PATH=./ext3/miniconda3/bin:$PATH
export PYTHONPATH=./ext3/miniconda3/bin:$PATH" > ./ext3/env.sh
source ./ext3/env.sh
conda update -n base conda -y
conda clean --all --yes
conda install pip
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install jupyter jupyterhub pandas matplotlib scipy scikit-learn scikit-image Pillow
pip install transformers sentencepiece datasets rouge_score tabulate
pip install tensorflow tensorboard
pip install --upgrade pip
pip install --upgrade "jax[cuda]==0.3.4" -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install flax optax
pip install GitPython
conda install -y git-lfs
pip install -r requirements.txt
python3 -m pip install -e src/