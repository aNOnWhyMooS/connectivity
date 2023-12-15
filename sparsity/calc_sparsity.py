# Calculate sparsity of a model as L2/L1 norm of the top head
import re
import sys, torch

from huggingface_hub import HfApi
from transformers import BertForSequenceClassification

model_repo = f"Jeevesh8/bert_ft_qqp_6ep-{sys.argv[1]}"
ckpts = {}
for commit in HfApi().list_repo_commits(model_repo):
    match_obj = re.match(r"Saving weights and logs of step (\d+)", commit.title.strip())
    if match_obj is not None:
        ckpts[int(match_obj.group(1))] = commit.commit_id

top_layer = BertForSequenceClassification.from_pretrained(model_repo, revision=ckpts[0]).classifier

with torch.no_grad():
    l2_norm = torch.norm(top_layer.weight.data)
    l1_norm = torch.norm(top_layer.weight.data, p=1)
    ratio = l2_norm/l1_norm
    bias_l2_norm = torch.norm(top_layer.bias.data)
    bias_l1_norm = torch.norm(top_layer.bias.data, p=1)

print(f"L2 norm for seed {sys.argv[1]} at step 0:", l2_norm.item())
print(f"L1 norm for seed {sys.argv[1]} at step 0:", l1_norm.item())
print(f"L2/L1 norm for seed {sys.argv[1]} at step 0:", ratio.item())
print(f"Bias for seed {sys.argv[1]} at step 0:", top_layer.bias.data)
print(f"L2 Norm of bias: {bias_l2_norm.item()}")
print(f"L1 Norm of bias: {bias_l1_norm.item()}")

import shutil
shutil.rmtree(local_dir)

# import re
# lis=[]
# for i in range(100):
#      with open(f"sbatch_outs/36027262_{i}.out") as f:
#         for line in f.readlines():
#             match_obj = re.fullmatch(r"L2/L1 norm for seed (\d+) at step 0: tensor\((.*)\)", line.strip())
#             if match_obj is None:
#                 continue
#             seed = int(match_obj.group(1))
#             ratio = float(match_obj.group(2))
#             assert seed==i, f"{seed}!={i}"
#             lis.append(ratio)
