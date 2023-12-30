import re
import time
from typing import Literal, List, Tuple, Optional

from huggingface_hub import HfApi
from requests.exceptions import ConnectionError
from .model_loaders.load_model import select_revision

def get_model_type(model: str) -> Literal['flax', 'tf', 'pytorch']:
    hf_api = HfApi()
    model_tags = hf_api.model_info(model).tags
    for e in ['pytorch', 'jax', 'tf']:
        if e in model_tags:
            if e=='jax':
                return 'flax'
            return e
    raise AssertionError(f"Can't determine type of model: {model}")

def steps_available(model: str, step: str) -> bool:
    try:
        select_revision(model, step)
        return True
    except ValueError as e:
        if 'Unable to find any commit' in str(e):
            return False
        raise e

def get_all_steps(model: str) -> List[int]:
    hf_api = HfApi()
    all_steps = []
    for commit in hf_api.list_repo_commits(model):
        match_obj = re.match(r'Saving weights and logs of step (\d+)', commit.title.strip())
        if match_obj is None:
            continue
        steps = match_obj.group(1)
        all_steps.append(steps)
    # Commits are already sorted by date, so no need to sort the steps.
    return all_steps

def get_step_pairs(model: str,) -> List[Tuple[str, str]]:
    all_steps = get_all_steps(model)
    return [(str(s1), str(s2))
            for i, s1 in enumerate(all_steps)
            for j, s2 in enumerate(all_steps)
            if i<j]

def get_all_models(substr: str, step: Optional[str] = None) -> List[str]:
    hf_api = HfApi()
    for _ in range(10):
        try:
            all_models = hf_api.list_models(search=substr)
            break
        except ConnectionError:
            time.sleep(3)
    
    if step is not None:
        models = [model.id for model in all_models
                  if steps_available(model.id, step)]
    else:
        models = all_models
    
    models = sorted(models)
    
    return models

def get_model_pairs(substr: str, step: Optional[str] = None) -> List[Tuple[str, str]]:

    models = get_all_models(substr, step)
    
    print(f'In total interpolating between {len(models)} models: {models}')

    return [(model1, model2)
            for i, model1 in enumerate(models) 
            for j, model2 in enumerate(models) if i<j]