#!/bin/bash
#SBATCH --job-name=finetune_berts
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_outs/%A_%a.out
#SBATCH --error=./sbatch_outs/%A_%a.err
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=22:59:59
#nvidia-smi

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif
OVERLAY_FILE=/scratch/$USER/finetune-jax/finetune-jax.ext3:ro

singularity exec --nv\
                 --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
                 /bin/bash -c "
source /ext3/env.sh
export HF_DATASETS_CACHE=\"/scratch/$USER/.cache/huggingface/datasets\"
export TRANSFORMERS_CACHE=\"/scratch/$USER/.cache/huggingface/transformers\"
export HF_METRICS_CACHE=\"/scratch/$USER/.cache/huggingface/metrics\"
python3 convert_to_pt.py $SLURM_ARRAY_TASK_ID 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 $1 $2 
exit
"
