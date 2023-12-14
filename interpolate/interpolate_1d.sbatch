#!/bin/bash
#SBATCH --job-name=interpolate_models
#SBATCH --open-mode=append
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=0-00:30:00
nvidia-smi

<<com
Submit a job for interpolating between a pair of models. 

Usage:
  To interpolate between a particular pair. The pair is specified in interpolate_1d.py's args:
    sbatch interpolate_1d.sbatch "<interpolate_1d.py args>"
  
  To interpolate between all models with a substr. The substr is specified in interpolate_1d.py's args:
    sbatch --array=0-<num_interps> interpolate_1d.sbatch "<interpolate_1d.py args>"
    
    <num_interps> is the total number of interpolations to perform -1. 
    
  Optionally, consider specifying --job-name/-J (before interpolate_1d.sbatch) to save .out files 
  with a particular name.
com

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
OVERLAY_FILE=/scratch/$USER/interpolate-torch/interpolate-torch.ext3:ro
singularity exec --nv\
                 --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
                 /bin/bash -c "
source /ext3/env.sh;
conda activate interpolate;
export HF_DATASETS_CACHE=\"/scratch/$USER/.cache/huggingface/datasets\"
export TRANSFORMERS_CACHE=\"/scratch/$USER/.cache/huggingface/transformers\"
export HF_METRICS_CACHE=\"/scratch/$USER/.cache/huggingface/metrics\"
if [[ -z ${SLURM_ARRAY_TASK_ID}]]; then
    python3 interpolate_1d.py $1;
else
    python3 interpolate_1d.py --job_id $SLURM_ARRAY_TASK_ID $1;
exit
"