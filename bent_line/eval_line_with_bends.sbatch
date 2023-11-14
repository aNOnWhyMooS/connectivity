#!/bin/bash
#SBATCH --job-name=bent_line_$2
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_outs/%A_%a.out
#SBATCH --error=./sbatch_outs/%A_%a.err
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:59:59
#SBATCH --mail-user=jeeveshjuneja@gmail.com
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
python3 -m pip install -e ../src
echo \"Base Model Prefix: $2\";
echo \"Dataset: $3\";

if [[ \"$3\" == \"qqp\" ]]; then
    python3 eval_line_with_bends.py --base_models_prefix $2 --base_model bert-base-uncased\
                                    --dataset $3 --models 00 02 03 01 --split validation\
                                    --num_exs 512 --experiment_id ${2//\//_}_$3;
elif [[ \"$3\" == \"paws\" ]]; then
    python3 eval_line_with_bends.py --base_models_prefix $2 --base_model bert-base-uncased\
                                    --dataset $3 --models 00 02 03 01 --split dev_and_test\
                                    --num_exs 512 --experiment_id ${2//\//_}_$3;

fi;

exit
"
