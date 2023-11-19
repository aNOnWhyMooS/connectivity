#!/bin/bash
#SBATCH --job-name=make_sbatch
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_outs/%j_%x.out
#SBATCH --error=./sbatch_outs/%j_%x.err
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=0-00:02:00
nvidia-smi

suffixes=""
half_saved_models="6 22 36 47 48 49 63"
for i in `seq 0 99`
do
    is_half_saved=false
    for j in $half_saved_models
    do
       if [ $j == $i ] ;
       then
          is_half_saved=true
       fi
    done
echo $is_half_saved
if [ ! $is_half_saved = true ]; then
    suffixes="${suffixes} ${i}"
fi;
done


SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif
OVERLAY_FILE=/scratch/$1/mode-conn-10/mode-conn-10.ext3:ro
singularity exec --nv\
		 --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
		 /bin/bash -c "
source /ext3/env.sh
export HF_DATASETS_CACHE=\"/scratch/$1/.cache/huggingface/datasets\"
export TRANSFORMERS_CACHE=\"/scratch/$1/.cache/huggingface/transformers\"
export HF_METRICS_CACHE=\"/scratch/$1/.cache/huggingface/metrics\"
mkdir ../../constellations//logs/$6/$2_interpol@$4steps/
echo $suffixes
python3 maker_runner.py --all_suffixes $suffixes\
	                    --base_models_prefix $7 --save_file ../../constellations/logs/$6/$2_permuted_interpol@$4steps/interpol_$2_$3.pkl\
						--dataset $2 --split $3 --num_steps $4 --free_overlays_begin $5 --user_name $1 --log_dir $6/$2_permuted_interpol@$4steps\
						--from_model_type $8 --base_model $9 --permute_wts
exit
"
