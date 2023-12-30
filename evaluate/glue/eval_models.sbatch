#!/bin/bash
#SBATCH --job-name=eval_models
#SBATCH --open-mode=append
#SBATCH --output=%A_%x_%a.out
#SBATCH --error=%A_%x_%a.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=0-06:30:00
nvidia-smi

<<com
Submit a job for evaluating models. 

Usage:
  To evaluate a particular model. The model and optionally a step is specified in eval_models.py's args:
    sbatch eval_models.sbatch "<eval_models.py args>"
  
  To evaluate all models with a substr. The substr is specified in eval_models.py's args:
    sbatch --array=0-<num_evals> eval_models.sbatch "<eval_models.py args>"
    
    <num_evals> is the total number of evaluations to perform -1. 
    
    Specifying '--step all' in eval_models.py's arguments will evaluate all models at all steps
    available for them.

  Optionally, consider specifying --job-name/-J (before eval_models.sbatch) to save .out files 
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
    python3 eval_models.py $1;
else
    python3 eval_models.py --job_id $SLURM_ARRAY_TASK_ID $1;
fi;
exit
"