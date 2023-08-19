#!/bin/bash
#SBATCH --job-name=svm
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_outs/%A_%a.out
#SBATCH --error=./sbatch_outs/%A_%a.err
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=0-00:05:59
#nvidia-smi

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif
OVERLAY_FILE=/scratch/$USER/mode-conn-0/mode-conn-0.ext3:ro

STEPS=$1

singularity exec --nv\
                 --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
                 /bin/bash -c "
source /ext3/env.sh
echo \"Running for ckpts at step: ${STEPS}\"
export HF_DATASETS_CACHE=\"/scratch/$USER/.cache/huggingface/datasets\"
export TRANSFORMERS_CACHE=\"/scratch/$USER/.cache/huggingface/transformers\"
export HF_METRICS_CACHE=\"/scratch/$USER/.cache/huggingface/metrics\"
python3 full_data_sep_plane.py ${SLURM_ARRAY_TASK_ID} ${STEPS}
"
