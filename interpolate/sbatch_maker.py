sbatch_format = """#!/bin/bash
#SBATCH --job-name=interpolate_models
#SBATCH --open-mode=append
#SBATCH --output=../../constellations/logs/{3}/%j_%x.out
#SBATCH --error=../../constellations/logs/{3}/%j_%x.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=0-08:59:59
nvidia-smi

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif
OVERLAY_FILE=/scratch/{2}/mode-conn-10/mode-conn-10.ext3:ro
singularity exec --nv\\
                 --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \\
                 /bin/bash -c "
source /ext3/env.sh; #{0}
#export XLA_PYTHON_CLIENT_PREALLOCATE=false;
export HF_DATASETS_CACHE=\\"/scratch/{2}/.cache/huggingface/datasets\\"
export TRANSFORMERS_CACHE=\\"/scratch/{2}/.cache/huggingface/transformers\\"
export HF_METRICS_CACHE=\\"/scratch/{2}/.cache/huggingface/metrics\\"
export TOKENIZERS_PARALLELISM=false;
#pip3 install git+https://github.com/Jeevesh8/trfrmr_weight_matching#subdirectory=src
#python3 -m pip install -e ../src/
{1}
exit
"
"""

import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cmds",
        type=str,
        nargs="+",
        required=True
    )

    parser.add_argument(
        "--overlay_file_nums",
        type=str,
        nargs="+",
        required=True,
    )
    
    parser.add_argument(
        "--sbatch_file_prefix",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--user_name",
        type=str,
        default="rb5139",
    )
    
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory within the logs directory,\
            where experiment results are to be logged."
    )
    return parser

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    assert len(args.overlay_file_nums)==len(args.cmds)
    for overlay_file_num, cmd in zip(args.overlay_file_nums, args.cmds):
        with open(args.sbatch_file_prefix+overlay_file_num+".sbatch", "w+") as f:
              f.write(sbatch_format.format(overlay_file_num, cmd, args.user_name, args.log_dir.strip("/")))
