# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install sentencepiece transformers datasets requests

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install GitPython

from transformers import BertForSequenceClassification, BertTokenizer

from git import Repo
from huggingface_hub import Repository

import torch
import sys
import string, random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""### Get Commits on Repo"""

model_repo = f"Jeevesh8/bert_ft_qqp_6ep-{sys.argv[1]}"
ran_string = ''.join(random.choices(string.ascii_lowercase, k=20))
local_dir = f"model_commits_{ran_string}/"
repo = Repository(local_dir=local_dir, clone_from=model_repo, skip_lfs_files=True)

repo = Repo(local_dir)

import re

ckpts = {}
for commit in repo.iter_commits("main"):
    match_obj = re.match(r"Saving weights and logs of step (\d+)", commit.message)
    if match_obj is not None:
        ckpts[int(match_obj.group(1))] = str(commit)

tokenizer = BertTokenizer.from_pretrained(model_repo)

"""### Get Data"""

from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

def get_qqp_data():
    dataset = load_dataset("glue", "qqp")
    dataset = dataset["validation"]
    datasets = []
    for i in range(len(dataset.features["label"].names)):
        datasets.append(dataset.shuffle(seed=42).filter(lambda e: e["label"]==i).select(list(range(512))))
    dataset = concatenate_datasets(datasets)
    return dataset

def get_paws_data():
    dataset = load_dataset("csv", data_files={"dev_and_test": "../paws_final/dev_and_test.tsv"}, delimiter="\t")["dev_and_test"]
    dataset = dataset.rename_columns({"sentence1": "question1", "sentence2": "question2"})
    return dataset

def tokenize_dataset(dataset):
    dataset = dataset.map(lambda e: tokenizer(e['question1'], e['question2'],
                                    truncation=True, padding='max_length',
                                    return_tensors='pt',),)

    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask',
                                            'token_type_ids', 'label'])

    dataset = dataset.map(lambda e1, e2, e3: {'input_ids': e1[0],
                                            'attention_mask': e2[0],
                                            'token_type_ids': e3[0]},
                            input_columns=['input_ids', 'attention_mask',
                                            'token_type_ids'])

    loader = DataLoader(dataset, batch_size=32)
    return loader

loaders = {"qqp" : tokenize_dataset(get_qqp_data()),
           "paws": tokenize_dataset(get_paws_data())}


"""### Calculate Embeddings"""

def shift_data(inp):
    for k, v in inp.items():
        inp[k] = v.to(device)
    return inp

def calc_embeddings(model, loader):
    embeddings = {}
    for i, batch in enumerate(loader):
        batch = shift_data(batch)
        batch["labels"] = batch.pop("label")
        out = model(**batch)
        for logits, input_ids in zip(out.logits, batch["input_ids"]):
            embeddings[tokenizer.decode(input_ids[input_ids!=tokenizer.pad_token_id])] = logits.cpu()
    return embeddings

import pickle

steps = int(sys.argv[3])
model = BertForSequenceClassification.from_pretrained(model_repo, revision=ckpts[steps]).to(device)
embeddings = {k : calc_embeddings(model, v) for k,v in loaders.items()}

with open(sys.argv[2], "wb") as f:
    pickle.dump(embeddings, f)

"""### Delete the model repo, to free up space"""
import shutil
shutil.rmtree(local_dir)