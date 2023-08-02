import sys, os, re
from huggingface_hub import HfApi

api = HfApi(token=sys.argv[1])

repo_id = "Jeevesh8/seed_step_wise_cosine_sims_bert_ft_qqp_6ep"
api.create_repo(repo_id=repo_id)

for f in os.listdir("./"):
    if f.endswith(".pkl"):
        match_obj = re.fullmatch(r"sample_wise_cosine_sims_(\d+)_(\d+).pkl", f)
        if match_obj is None:
            continue
        seed, steps = match_obj.group(1), match_obj.group(2)
        api.upload_file(f, repo_id=repo_id, 
                        commit_message=f"Adding cosine similarity of gradients on 1024 samples from qqp-validation for model Jeevesh8/bert_ft_qqp_6ep-{seed} at steps {steps}")