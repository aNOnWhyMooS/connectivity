#!/bin/bash
#SBATCH --job-name=make_sbatch
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_outs/%j_%x.out
#SBATCH --error=./sbatch_outs/%j_%x.err
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=0-00:02:00
nvidia-smi

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif
OVERLAY_FILE=/scratch/$1/mode-conn-1/mode-conn-1.ext3
singularity exec --nv\
		 --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
		 /bin/bash -c "
source /ext3/env.sh
export HF_DATASETS_CACHE=\"/scratch/$1/.cache/huggingface/datasets\"
export TRANSFORMERS_CAHCE=\"/scratch/$1/.cache/huggingface/transformers\"
export HF_METRICS_CACHE=\"/scratch/$1/.cache/huggingface/metrics\"
python3 maker_runner.py --all_ft_seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47\
	                --pt_model_name_or_path google/multiberts-seed_0-step_$2 --models_save_dir mnli_ft/ --free_overlays_begin $3 --user_name $1
exit
"
