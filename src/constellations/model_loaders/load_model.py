import functools, os, re
import shutil
from typing import List, Optional, Type
from urllib.error import HTTPError

import torch
from git import Repo
from huggingface_hub import Repository, HfApi
from transformers import AutoModelForSequenceClassification, AutoModel
import warnings

@functools.lru_cache(125)
def select_revision(path_or_name, num_steps: int, local_dir=None, tmp_dir=None):
    """Return the latest commit with num_steps in its commit message."""
    import string, random
    
    if num_steps is None:
        return None
    
    if tmp_dir is None:
        tmp_dir = "."+''.join(random.choices(string.ascii_uppercase+string.digits, k=20))
    if local_dir is not None:
        repo = Repo(local_dir)
    else:
        try:
            shutil.rmtree(tmp_dir)
        except FileNotFoundError:
            print(f"Creating {tmp_dir}, for loading in git data")
        while True:
            try:
                repo = Repository(local_dir=tmp_dir, clone_from=path_or_name, 
                                  skip_lfs_files=True)
                break
            except OSError as e:
                if "Cloning into" not in str(e):
                    raise e
        repo = Repo(tmp_dir)
    
    for commit in repo.iter_commits("main"):
        if f" {num_steps} steps" in commit.message:
            selected_commit=str(commit)
            break
    else:
        raise ValueError(f"Unable to find any commit with {num_steps} steps")
    
    return selected_commit

def get_model(path_or_name,
              base_model: Optional[str]="bert-base-uncased",
              from_flax: bool = False,
              model_type: Type[AutoModel]=AutoModel,
              **select_revision_kwargs) -> torch.nn.Module:
    """Returns a sequence classification model loaded from path_or_name. If it can't load 
    the model from path_or_name, it treats the path_or_name as a state dict file, and tries 
    to load it using torch.load() after appending ".pt" to it. 
    
    NOTE: Wraps the forward() call to accept a single input.
    """
    if len(select_revision_kwargs)!=0:
        if os.path.isdir(path_or_name):
            raise AssertionError(f"To load older commits, must fetch model from HuggingFace remote url repo,\
                but {path_or_name} is local directory!")
        revision = select_revision(path_or_name, **select_revision_kwargs)
    else:
        revision=None #===latest

    try:    
        model = model_type.from_pretrained(path_or_name, from_flax=from_flax, revision=revision)
    except (HTTPError, OSError, ValueError) as e:
        print("Encountered Error:", e, flush=True)
        print("Trying to load model from {}.pt".format(path_or_name), flush=True)
        model = model_type.from_pretrained(base_model, num_labels=3,)
        state_dict_path = path_or_name+".pt"
        try:
            model.load_state_dict(torch.load(state_dict_path))
            print("Model loaded from {}.pt".format(path_or_name), flush=True)
        except:
            if "does not appear to have a file named flax_model.msgpack." in str(e):
                try:
                    model = model_type.from_pretrained(path_or_name, revision=revision)
                    print(f"PyTorch model {path_or_name}, loaded from HuggingFace hub.")
                except:
                    pass
            else:
                raise e
        
        
    def decorator(func):
        @functools.wraps(func)
        def wrapper(X):
            return func(**X)
        return wrapper
    
    model.forward = decorator(model.forward)
    return model

get_sequence_classification_model = functools.partial(get_model,model_type=AutoModelForSequenceClassification)

def get_models_ft_with_prefix(prefix: str, only_digit_suffix:bool=True) -> List[str]:
    hf_api = HfApi()
    models = hf_api.list_models(search=prefix)
    if only_digit_suffix:
        final_models = []
        for model in models:
            if re.fullmatch(r"\d+", model.modelId[len(prefix):]) is not None:
                final_models.append(model.modelId)
            else:
                warnings.warn(f"Skipping model with non-digit suffix \
                    {model} for provided prefix: {prefix}. Use \
                    only_digit_suffix=False for including these models too.")
    else:
        final_models = models
    
    return final_models
                
    
