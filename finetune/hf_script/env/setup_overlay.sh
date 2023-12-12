<<usage
Creates an overlay for finetuning using huggingface script.
Usage:
  bash setup_overlay.sh [path to packages.sh]

packages.sh should contain commands to install packages in 
a conda environment.

If path not specified, will try to use the packages.sh in the
directory from where the command is run.
usage

#!/bin/bash
set -e

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
OVERLAY=finetune-jax/finetune-jax.ext3

packs="$(pwd)/packages.sh"
packages=${1:-${packs}}

cd $SCRATCH

mkdir finetune-jax

cp -rp /scratch/work/public/overlay-fs-ext3/overlay-25GB-500K.ext3.gz ./finetune-jax/

gunzip finetune-jax/overlay-25GB-500K.ext3.gz

mv finetune-jax/overlay-25GB-500K.ext3 finetune-jax/finetune-jax.ext3

singularity exec --overlay $OVERLAY $SINGULARITY_IMAGE /bin/bash -c "
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
echo \"source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:\$PATH
export PYTHONPATH=/ext3/miniconda3/bin:\$PATH\" > /ext3/env.sh

source /ext3/env.sh
conda update -n base conda -y
conda clean --all --yes

conda create -n finetune python=3.10 -y
conda activate finetune

bash ${packages};
" 
