<<com
Submit a job for interpolating between a pair of models. 

Usage:
  sbatch interpolate_1d.sbatch <out_dir> "<interpolate_1d.py args>"

com

#!/bin/bash
#SBATCH --job-name=interpolate_models
#SBATCH --open-mode=append
#SBATCH --output=$1/%j_%x.out
#SBATCH --error=$1/%j_%x.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=0-08:59:59
nvidia-smi

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
python3 interpolate_1d.py $2;
exit
"