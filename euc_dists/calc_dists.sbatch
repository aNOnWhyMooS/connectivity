#!/bin/bash
#SBATCH --job-name=interpolate_models
#SBATCH --open-mode=append
#SBATCH --output=%A_%x_%a.out
#SBATCH --error=%A_%x_%a.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=128G
#SBATCH --time=0-00:05:00
nvidia-smi

<<com
Submit a job for calculating euclidean distance between a pair of models.

Usage:
  To interpolate between a particular pair. The pair is specified in calc_dists.py's args:
    sbatch calc_dists.sbatch "<calc_dists.py args>"
  
  To interpolate between all models with a substr. The substr is specified in calc_dists.py's args:
    sbatch --array=0-<num_interps> calc_dists.sbatch "<calc_dists.py args>"
    
    <num_interps> is the total number of interpolations to perform -1. 
    
  Optionally, consider specifying --job-name/-J (before calc_dists.sbatch) to save .out files 
  with a particular name.
com

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
OVERLAY_FILE=/scratch/$USER/interpolate-torch/interpolate-torch.ext3:ro
singularity exec --nv\
                 --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
                 /bin/bash -c "
source /ext3/env.sh;
conda activate interpolate;
export HF_DATASETS_CACHE=\"/scratch/$USER/.cache/huggingface/datasets\";
export TRANSFORMERS_CACHE=\"/scratch/$USER/.cache/huggingface/transformers\";
export HF_METRICS_CACHE=\"/scratch/$USER/.cache/huggingface/metrics\";
if [[ -z \"${SLURM_ARRAY_TASK_ID}\" ]]; then
    python3 calc_dists.py $1;
else
    python3 calc_dists.py --job_id $SLURM_ARRAY_TASK_ID $1;
fi;
exit
"