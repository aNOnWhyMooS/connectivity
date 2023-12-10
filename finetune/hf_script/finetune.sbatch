#!/bin/bash
#SBATCH --job-name=finetune_models
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_outs_%A_%a.out
#SBATCH --error=./sbatch_outs_%A_%a.err
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1-23:59:59
#nvidia-smi

<<usage
Finetunes a sequence classification model.
Usage:
  sbatch --array=0-10 finetune.sbatch <hub_token> <pretrained model>\
                                      <output_dir_prefix> <dataset>
usage


SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
OVERLAY_FILE=/scratch/$USER/finetune-jax/finetune-jax.ext3:ro

singularity exec --nv\
                 --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
                 /bin/bash -c "
source /ext3/env.sh
conda activate finetune
export HF_DATASETS_CACHE=\"/scratch/$USER/.cache/huggingface/datasets\"
export TRANSFORMERS_CACHE=\"/scratch/$USER/.cache/huggingface/transformers\"
export HF_METRICS_CACHE=\"/scratch/$USER/.cache/huggingface/metrics\"
echo \"Hub Token: $1\"
echo \"Task name: $4\"
echo \"Pretrained model: $2\"
python run_flax_glue.py \
        --model_name_or_path $2 \
        --task_name $4 \
        --max_seq_length 128 \
        --learning_rate 2e-5 \
        --num_train_epochs 4 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 512 \
        --eval_steps 2000 --save_steps 2000 --logging_steps 2000\
        --output_dir $3_$4_ft_${SLURM_ARRAY_TASK_ID}/ \
        --seed ${SLURM_ARRAY_TASK_ID} --push_to_hub\
        --hub_token $1 --warmup_steps 3411 --weight_decay 0.01
exit
"
