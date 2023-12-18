<<com
Installs packages necessary for original BERT scripts's finetuning.
com

#!/bin/bash
set -e
conda install tensorflow-gpu==1.15.0 -y
conda install numpy==1.19.5 -y
pip install getgist -y
getgist raffaem download_glue_data.py
python3 download_glue_data.py --data_dir glue_data --tasks QQP
python3 download_glue_data.py --data_dir glue_data --tasks MNLI
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
rm uncased_L-12_H-768_A-12.zip