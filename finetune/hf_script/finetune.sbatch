#!/bin/bash
#SBATCH --job-name=finetune_models
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_outs/%A_%a.out
#SBATCH --error=./sbatch_outs/%A_%a.err
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1-23:59:59
#nvidia-smi

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif
OVERLAY_FILE=/scratch/$USER/mode-conn-0/mode-conn-0.ext3:ro

singularity exec --nv\
                 --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
                 /bin/bash -c "
source /ext3/env.sh
export HF_DATASETS_CACHE=\"/scratch/$USER/.cache/huggingface/datasets\"
export TRANSFORMERS_CACHE=\"/scratch/$USER/.cache/huggingface/transformers\"
export HF_METRICS_CACHE=\"/scratch/$USER/.cache/huggingface/metrics\"

python run_flax_glue.py \
        --model_name_or_path $1 \
        --task_name $3 \
        --max_seq_length 128 \
        --learning_rate 2e-5 \
        --num_train_epochs 6 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 512 \
        --eval_steps 10000 --save_steps 2000\
        --output_dir $2_$3_ft_${SLURM_ARRAY_TASK_ID}/ \
        --seed ${SLURM_ARRAY_TASK_ID} --push_to_hub\
        --hub_token $0 --warmup_steps 3411 --weight_decay 0.01
exit
"
