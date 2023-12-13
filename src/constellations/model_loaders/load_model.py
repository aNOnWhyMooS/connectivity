import functools, os, re
import shutil
from typing import List, Optional, Type, Literal
from urllib.error import HTTPError

import torch
from git import Repo
from huggingface_hub import Repository, HfApi
from transformers import AutoModelForSequenceClassification, AutoModel, FlaxAutoModelForSequenceClassification
import warnings

@functools.lru_cache(125)
def select_revision(path_or_name: str, num_steps: int|str,):
    """Return the latest commit with num_steps in its commit message."""
    import string, random

    if num_steps is None:
        return None

    tmp_dir = "_"+''.join(random.choices(string.ascii_uppercase+string.digits, k=20))
    try:
        shutil.rmtree(tmp_dir)
    except FileNotFoundError:
        pass
    print(f"Creating {tmp_dir}, for loading in git data")
    repo = Repository(local_dir=tmp_dir, clone_from=path_or_name,
                        skip_lfs_files=True)
    repo = Repo(tmp_dir)

    for commit in repo.iter_commits("main"):
        if f"Saving weights and logs of step {num_steps}" == commit.message.strip():
            selected_commit=str(commit)
            break
    else:
        raise ValueError(f"Unable to find any commit with {num_steps} steps")
    shutil.rmtree(tmp_dir)
    return selected_commit

def get_model(path_or_name,
              base_model: Optional[str]="bert-base-uncased",
              from_flax: bool = False,
              model_type: Type[AutoModel]=AutoModel,
              ret_type: Literal['flax', 'pt'] = 'pt',
              wrap_forward: bool = True,
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
        revision = None #===latest

    try:
        if ret_type=='pt':
            model = model_type.from_pretrained(path_or_name, from_flax=from_flax, revision=revision)
        elif ret_type=='flax':
            model = model_type.from_pretrained(path_or_name, from_pt=(not from_flax), revision=revision)
        else:
            raise ValueError(f'unknown ret type: {ret_type}. Should be pt or flax.')

    except (HTTPError, OSError, ValueError) as e:
        print("Encountered Error:", e, flush=True)
        print("Trying to load model from {}.pt".format(path_or_name), flush=True)

        state_dict_path = path_or_name+".pt"
        if ret_type=='flax':
            raise NotImplementedError('Can\'t load flax model from .pt file.')
        params = torch.load(state_dict_path)
        model = model_type.from_pretrained(base_model, num_labels=params['classifier.weight'].shape[0],)

        try:
            model.load_state_dict(params)
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

    if wrap_forward:
        assert ret_type=='pt', 'forward method only in pt models, so can only wrap them'
        model.forward = decorator(model.forward)
    return model

get_sequence_classification_model = functools.partial(get_model,model_type=AutoModelForSequenceClassification)
get_flax_seq_classification_model = functools.partial(get_model,
                                                      model_type=FlaxAutoModelForSequenceClassification,
                                                      ret_type='flax', wrap_forward=False)

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


