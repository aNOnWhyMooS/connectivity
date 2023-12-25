import re
import glob
import pickle
import argparse
from typing import Literal, List

import time
import torch
from huggingface_hub import HfApi
from requests.exceptions import ConnectionError
from match_finder import match_params

from constellations.model_loaders.load_model import get_sequence_classification_model, get_flax_seq_classification_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_type(model: str) -> Literal['flax', 'tf', 'pytorch']:
    hf_api = HfApi()
    model_tags = hf_api.model_info(model).tags
    for e in ['pytorch', 'jax', 'tf']:
        if e in model_tags:
            if e=='jax':
                return 'flax'
            return e
    raise AssertionError(f"Can't determine type of model: {model}")

def get_all_steps(model: str):
    hf_api = HfApi()
    all_steps = []
    for commit in hf_api.list_repo_commits(model):
        match_obj = re.match(r'Saving weights and logs of step (\d+)', commit.title.strip())
        if match_obj is None:
            continue
        steps = match_obj.group(1)
        all_steps.append(steps)
    return all_steps

def get_all_models(substr: str):
    for _ in range(10):
        try:
            all_models = hf_api.list_models(search=substr)
            return all_models
        except ConnectionError:
            time.sleep(3)

def all_steps_available(models: List[str]):
    model_wise_steps = {}
    for model in models:
        model_wise_steps[model] = get_all_steps(model)
    
    step_wise_models = {}
    for model, steps in model_wise_steps.items():
        for step in steps:
            if step not in step_wise_models:
                step_wise_models[step] = []
            step_wise_models[step].append(model)
    
    return model_wise_steps, step_wise_models

def get_model_across_steps_pairs(model: str, span: int):
    all_steps = get_all_steps(model)
    all_steps = sorted([int(e) for e in all_steps])
    return [(model, model), (s1, s2)
            for i, s1 in enumerate(all_steps)
            for j, s2 in enumerate(all_steps)
            if j-span<=i<j]


def get_model_at_steps_pairs(substr: str, step: str):
    
    _, step_wise_models = all_steps_available(get_all_models(substr))
    
    if step is None:
        # Generate euc distance pairs for all step values.
        model_at_step_pairs = []
        for step, models in sorted(step_wise_models.items()):
            model_at_step_pairs += [((m1, m2), (step, step)) 
                                    for i, m1 in enumerate(models)
                                    for j, m2 in enumerate(models)
                                    if i<j]
    else:
        # Generate euc distance pairs for only required step.
        model_at_step_pairs = [((m1, m2), (step, step)) 
                                for i, m1 in enumerate(step_wise_models[step])
                                for j, m2 in enumerate(step_wise_models[step])
                                if i<j]
    
    return model_at_step_pairs

def main(args):
    model1_kwargs = {'path_or_name': args.models[0], 
                     'from_flax' : (args.from_model_type=="flax"),
                     'num_steps' : args.steps[0], }
    
    model2_kwargs = {'path_or_name': args.models[1], 
                     'from_flax' : (args.from_model_type=="flax"),
                     'num_steps' : args.steps[1], }

    if args.do_perm:
        m1 = get_flax_seq_classification_model(**model1_kwargs)
        m2 = get_flax_seq_classification_model(**model2_kwargs)
        m1, m2 = match_params(m1, m2, model_type = (
            'roberta' if 'roberta' in args.base_model else 'bert'))
        w1, w2 = m1.state_dict(), m2.state_dict()
    else:
        w1 = get_sequence_classification_model(**model1_kwargs).state_dict()
        w2 = get_sequence_classification_model(**model2_kwargs).state_dict()

    euclidean_dist = torch.sqrt(sum([torch.sum((v1-v2)*(v1-v2)) for (_, v1), (_, v2) in zip(w1.items(), w2.items())])).item()
    print(f"Euclidean distance between {args.models[0]}@{args.steps[0]} and {args.models[1]}@{args.steps[1]}: {euclidean_dist}.", flush=True)
    return euclidean_dist

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Interpolate between model pairs.")

    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma separated list of models to calculate euclidean distance between. "
        "Or a substring. Distance will be calculated for all models on hf-hub having "
        "this substring. Or a single model, in which case distance will be calculated "
        "between the various checkpoints of that model throughout training.",
    )

    parser.add_argument(
        "--save_file",
        type=str,
        required=True,
        help="Name of file where to save the euclidean distance value,"
        "using pickle."
    )

    parser.add_argument(
        "--steps",
        type=str,
        help="Comma separated pair of steps at which to fetch "
            "the two models specified in args.models. "
            "A commit with this number of ' \d+ steps' in its "
            "commit message, must be present on the remote. By default, latest "
            "model will be loaded. Can also be a single number in case both "
            "numbers are supposed to be same.",
    )

    parser.add_argument(
        "--do_perm",
        action="store_true",
        help="If specified, permutation will be done, before"
            "interpolating between models."
    )

    parser.add_argument(
        '--job_id',
        required=False,
        type=int,
        help="In case args.models is a substring, or a single model, this tells "
        "which exact pair of models to interpolate between."
    )

    parser.add_argument(
        '--span',
        required=False,
        type=int,
        default=1,
        help='Used when calc. distance between checkpoints of the same model.'
        'Each ckpt\'s distance from span ckpts before and after it will be '
        'calculated. (Default: 1)'
    )

    args = parser.parse_args()

    vals_dict = {}

    hf_api = HfApi()

    if args.steps is None:
        models = args.models.split(',')
        if len(models)>1:
            # Measure between latest checkpoints
            args.steps = (None, None)
            args.models = models
        elif hf_api.repo_exists(models[0]):
            # Measure for all checkpoints within this model.
            args.models, args.steps = get_model_across_steps_pairs(models[0], args.span)[args.job_id]
        else:
            # Measure (between every pair of models) at every step.
            args.models, args.steps = get_model_at_steps_pairs(models[0],)[args.job_id]
    
    elif len(args.models.split(','))==1:
        models = args.models.split(',')
        steps = args.steps.split(',')
        if len(steps)==1:
            # Measure (between every pair of models) at this step.
            args.models, args.steps = get_model_at_steps_pairs(models[0], args.steps[0])[args.job_id]
        else:
            # Measure for the same model at two different steps.
            args.steps = steps
            args.models = (models[0], models[0])
    
    elif len(args.models.split(','))==2:
        models = args.models.split(',')
        steps = args.steps.split(',')
        if len(steps)==1:
            # Measure distance for two models at this step. 
            args.models = models
            steps = (steps[0], steps[0])
        else:
            # Measure distance between two models taken at the specified steps for each model.
            args.steps = steps
            args.models = models
    else:
        raise ValueError('Check passed args:', args)

    if args.job_id is not None:
        suffix = args.save_file.split('.')[-1]
        prefix = ('.'.join(args.save_file.split('.')[:-1])
                  +f'_{args.models[0].split("/")[-1]}@{args.steps[0]}_{args.models[1].split("/")[-1]}@{args.steps[1]}')

        already_completed = glob.glob(f'{prefix}_*.{suffix}')
        if already_completed:
            print(f'Job already completed with logs at:', already_completed, '. Exitting.')
            exit(0)

        args.save_file = f'{prefix}_{args.job_id}.{suffix}'
    args.experiment_id = args.save_file.replace('/', '_')

    args.from_model_type = get_model_type(args.models[0])
    assert args.from_model_type==get_model_type(args.models[1])

    euclidean_dist = main(args)
    print(f"Calculated euc distance from {args.models[0]}@{args.steps[0]} to {args.models[1]}@{args.steps[1]}",
          flush=True)
    vals_dict[((args.models[0], args.steps[0]), (args.models[1], args.steps[1]))] = euclidean_dist
    vals_dict[((args.models[1], args.steps[1]), (args.models[0], args.steps[0]))] = euclidean_dist

    with open(args.save_file, "wb") as f:
        pickle.dump(vals_dict, f)

    print(f"Wrote the values to {args.save_file}!", flush=True)
