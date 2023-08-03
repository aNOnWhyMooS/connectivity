# Calculate sparsity of a model as L2/L1 norm of the top head
import sys, torch
import random, string
from git import Repo

from huggingface_hub import Repository
from transformers import BertForSequenceClassification

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

top_layer = BertForSequenceClassification.from_pretrained(model_repo, revision=ckpts[0]).classifier

with torch.no_grad():
    l2_norm = torch.norm(top_layer.weight.data)
    l1_norm = torch.norm(top_layer.weight.data, p=1)
    ratio = l2_norm/l1_norm
    bias_l2_norm = torch.norm(top_layer.bias.data)
    bias_l1_norm = torch.norm(top_layer.bias.data, p=1)

print(f"L2/L1 norm for seed {sys.argv[1]} at step 0:", ratio)
print(f"Bias for seed {sys.argv[1]} at step 0:", top_layer.bias.data)
print(f"L2 Norm of bias: {bias_l2_norm}")
print(f"L1 Norm of bias: {bias_l1_norm}")