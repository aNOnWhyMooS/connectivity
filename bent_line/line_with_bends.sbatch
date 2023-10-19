#!/bin/bash
#SBATCH --job-name=bent_line_$2
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_outs/%A_%a.out
#SBATCH --error=./sbatch_outs/%A_%a.err
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=1-23:59:59
#nvidia-smi

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif
OVERLAY_FILE=/scratch/${USER}/mode-conn-$1/mode-conn-$1.ext3:ro


singularity exec --nv\
             --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
             /bin/bash -c "
source /ext3/env.sh
export HF_DATASETS_CACHE=\"/scratch/${USER}/.cache/huggingface/datasets\"
export TRANSFORMERS_CACHE=\"/scratch/${USER}/.cache/huggingface/transformers\"
export HF_METRICS_CACHE=\"/scratch/${USER}/.cache/huggingface/metrics\"
python3 -m pip install -e ../../../connectivity/src
python3 line_with_bends.py --base_models_prefix Jeevesh8/$3 --base_model bert-base-uncased\
                           --dataset qqp --batch_size 2 --gradient_accumulation_steps 4\
                           --n_sample 4 --break_epochs 10\
                           --num_steps $4 --indices $5;
exit
"
