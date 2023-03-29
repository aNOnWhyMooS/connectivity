#!/bin/bash
#SBATCH --job-name=eval_berts_$3
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_outs/%A_%a.out
#SBATCH --error=./sbatch_outs/%A_%a.err
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=1-23:59:59
#nvidia-smi

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif
OVERLAY_FILE=/scratch/$1/mode-conn-$2/mode-conn-$2.ext3:ro


if [[ $3 = "hans" ]]
then
    singularity exec --nv\
                 --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
                 /bin/bash -c "
    source /ext3/env.sh
    export HF_DATASETS_CACHE=\"/scratch/$1/.cache/huggingface/datasets\"
    export TRANSFORMERS_CACHE=\"/scratch/$1/.cache/huggingface/transformers\"
    export HF_METRICS_CACHE=\"/scratch/$1/.cache/huggingface/metrics\"
    python3 -m pip install -e ../../../connectivity/src
    mkdir ../../../constellations/logs/NLI/$6/hans_eval@$5steps/
    python3 eval_models.py --base_models_prefix Jeevesh8/$4\
                        --dataset hans --split ${7:-test} --all_data\
                        --write_file_prefix ../../../constellations/logs/NLI/$6/hans_eval@$5steps/$4\
                        --num_steps $5 --base_model bert-base-uncased --from_model_type pt;
    exit
    "
elif [[ $3 = "mnli" ]]
then
    singularity exec --nv\
                 --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
                 /bin/bash -c "
    source /ext3/env.sh
    export HF_DATASETS_CACHE=\"/scratch/$1/.cache/huggingface/datasets\"
    export TRANSFORMERS_CACHE=\"/scratch/$1/.cache/huggingface/transformers\"
    export HF_METRICS_CACHE=\"/scratch/$1/.cache/huggingface/metrics\"
    python3 -m pip install -e ../../../connectivity/src
    mkdir ../../../constellations/logs/NLI/$6/mnli_eval@$5steps/
    python3 eval_models.py --base_models_prefix Jeevesh8/$4\
                        --dataset mnli --split ${7:-test}\
                        --write_file_prefix ../../../constellations/logs/NLI/$6/mnli_eval@$5steps/$4\
                        --num_steps $5 --base_model bert-base-uncased --from_model_type pt;
    exit
    "
elif [[ $3 = "cola" ]]
then
    singularity exec --nv\
                 --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
                 /bin/bash -c "
    source /ext3/env.sh
    export HF_DATASETS_CACHE=\"/scratch/$1/.cache/huggingface/datasets\"
    export TRANSFORMERS_CACHE=\"/scratch/$1/.cache/huggingface/transformers\"
    export HF_METRICS_CACHE=\"/scratch/$1/.cache/huggingface/metrics\"
    python3 -m pip install -e ../../../connectivity/src
    mkdir ../../../constellations/logs/NLI/$6/cola_eval@$5steps/
    python3 eval_models.py --base_models_prefix Jeevesh8/$4\
                        --dataset cola --split ${7:-validation}\
                        --write_file_prefix ../../../constellations/logs/NLI/$6/cola_eval@$5steps/$4\
                        --num_steps $5 --base_model bert-base-uncased --from_model_type pt;
    exit
    "
elif [[ $3 = "paws" ]]
then
    singularity exec --nv\
                 --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
                 /bin/bash -c "
    source /ext3/env.sh
    export HF_DATASETS_CACHE=\"/scratch/$1/.cache/huggingface/datasets\"
    export TRANSFORMERS_CACHE=\"/scratch/$1/.cache/huggingface/transformers\"
    export HF_METRICS_CACHE=\"/scratch/$1/.cache/huggingface/metrics\"
    python3 -m pip install -e ../../../connectivity/src
    mkdir ../../../constellations/logs/NLI/$6/paws_eval@$5steps/
    python3 eval_models.py --base_models_prefix Jeevesh8/$4\
                        --dataset paws --split ${7:-dev_and_test}\
                        --write_file_prefix ../../../constellations/logs/NLI/$6/paws_eval@$5steps/$4\
                        --num_steps $5 --base_model bert-base-uncased --from_model_type pt;
    exit
    "

elif [[ $3 = "qqp" ]]
then
    singularity exec --nv\
                 --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
                 /bin/bash -c "
    source /ext3/env.sh
    export HF_DATASETS_CACHE=\"/scratch/$1/.cache/huggingface/datasets\"
    export TRANSFORMERS_CACHE=\"/scratch/$1/.cache/huggingface/transformers\"
    export HF_METRICS_CACHE=\"/scratch/$1/.cache/huggingface/metrics\"
    python3 -m pip install -e ../../../connectivity/src
    mkdir ../../../constellations/logs/NLI/$6/qqp_eval@$5steps/
    python3 eval_models.py --base_models_prefix Jeevesh8/$4\
                        --dataset qqp --split ${7:-validation}\
                        --write_file_prefix ../../../constellations/logs/NLI/$6/qqp_eval@$5steps/$4\
                        --num_steps $5 --base_model bert-base-uncased --from_model_type pt;
    exit
    "

fi
