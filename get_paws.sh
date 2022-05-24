#!/bin/bash
wget http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv
mkdir quora_orig/
mv quora_duplicate_questions.tsv quora_orig/data.tsv
wget https://storage.googleapis.com/paws/english/paws_qqp.tar.gz
tar -xzf paws_qqp.tar.gz
git clone https://github.com/aNOnWhyMooS/paws/
cd paws; git checkout patch-1;
mkdir ../paws_final/
pip uninstall -y nltk
pip install nltk==3.2.5
python qqp_generate_data.py \
  --original_qqp_input="../quora_orig/data.tsv" \
  --paws_input="../paws_qqp/train.tsv" \
  --paws_output="../paws_final/train.tsv"
python qqp_generate_data.py \
  --original_qqp_input="../quora_orig/data.tsv" \
  --paws_input="../paws_qqp/dev_and_test.tsv" \
  --paws_output="../paws_final/dev_and_test.tsv"
pip uninstall -y nltk
pip install nltk
