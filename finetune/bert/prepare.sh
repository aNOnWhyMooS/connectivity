#!/bin/bash
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif
OVERLAY_FILE=/scratch/$1/mode-conn-$2/mode-conn-$2.ext3

singularity exec --nv --overlay $OVERLAY_FILE $SINGULARITY_IMAGE /bin/bash -c "
source /ext3/env.sh
pip install getgist
getgist raffaem download_glue_data.py
conda install python=3.7
conda install tensorflow-gpu==1.15.0
pip install numpy==1.19.5
python3 download_glue_data.py --data_dir glue_data --tasks QQP
python3 -c \"import tensorflow as tf; print(tf.__version__);\"
"