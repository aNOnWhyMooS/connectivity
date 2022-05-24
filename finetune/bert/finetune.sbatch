#!/bin/bash
#SBATCH --job-name=finetune_berts
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_outs/%A_%a.out
#SBATCH --error=./sbatch_outs/%A_%a.err
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=22:59:59
#nvidia-smi

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif
OVERLAY_FILE=/scratch/$USER/mode-conn-0/mode-conn-0.ext3:ro

BERT_BASE_DIR=./uncased_L-12_H-768_A-12
GLUE_DIR=./glue_data

singularity exec --nv\
                 --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
                 /bin/bash -c "
source /ext3/env.sh
export HF_DATASETS_CACHE=\"/scratch/$USER/.cache/huggingface/datasets\"
export TRANSFORMERS_CACHE=\"/scratch/$USER/.cache/huggingface/transformers\"
export HF_METRICS_CACHE=\"/scratch/$USER/.cache/huggingface/metrics\"
python3 run_classifier.py \
  --task_name=qqp \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/QQP \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=qqp_save_$SLURM_ARRAY_TASK_ID --save_checkpoints_steps=5000
exit
"
