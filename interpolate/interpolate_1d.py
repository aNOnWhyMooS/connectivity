import pickle
import argparse
from typing import Literal, List, Tuple

import tabulate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_metric
from transformers import AutoTokenizer
from huggingface_hub import HfApi
from match_finder import match_params

from constellations.model_loaders.modelling_utils import get_criterion_fn, get_logits_converter, get_pred_fn, linear_comb
from constellations.model_loaders.load_model import get_sequence_classification_model, get_flax_seq_classification_model, select_revision
from constellations.dataloaders.loader import get_loader
from constellations.utils.eval_utils import eval

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

def steps_available(model: str, step: str):
    try:
        select_revision(model, step)
        return True
    except ValueError as e:
        if 'Unable to find any commit' in str(e):
            return False
        raise e

def get_model_pairs(substr: str, step: str) -> List[Tuple[str, str]]:
    hf_api = HfApi()
    models = [model.id for model in hf_api.list_models(search=substr)
              if steps_available(model.id, step)]
    models = sorted(models)

    print(f'In total interpolating between {len(models)} models: {models}')

    return [(model1, model2)
            for i, model1 in enumerate(models) 
            for j, model2 in enumerate(models) if i<j]

def main(args):
    is_feather_bert = (("feather" in args.models[0])
                    or ("feather" in args.models[1]))

    if is_feather_bert:
        mnli_label_dict = {"contradiction": 0, "entailment": 1, "neutral": 2}
    else:
        mnli_label_dict = {"contradiction": 2, "entailment": 0, "neutral": 1}

    hans_label_dict = {"entailment": 0, "non-entailment": 1}

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    input_target_loader = get_loader(args, tokenizer, mnli_label_dict)

    ce_loss = nn.CrossEntropyLoss()

    mnli_logits_to_hans = get_logits_converter(mnli_label_dict, hans_label_dict)

    logit_converter = mnli_logits_to_hans if args.dataset=="hans" else None

    criterion = get_criterion_fn(ce_loss, logit_converter)
    pred_fn   = get_pred_fn(logit_converter_fn = logit_converter)

    if args.dataset=="cola":
        metric = load_metric("glue", "cola", experiment_id=args.experiment_id)
    elif args.dataset=="qqp":
        metric = load_metric("glue", "qqp", experiment_id=args.experiment_id)
    else:
        metric = load_metric("accuracy", experiment_id=args.experiment_id)

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

    model = get_sequence_classification_model(**model1_kwargs)

    euclidean_dist = torch.sqrt(sum([torch.sum((v1-v2)*(v1-v2)) for (_, v1), (_, v2) in zip(w1.items(), w2.items())])).item()
    print(f"Euclidean distance between {args.models[0]}@{args.steps[0]} and {args.models[1]}@{args.steps[1]}: {euclidean_dist}.", flush=True)
    print(f'Interpolating between {args.models[0]}@{args.steps[0]} and {args.models[1]}@{args.steps[1]} on {args.dataset}', flush=True)

    xy_min = (0,1)
    xy_max = (1,0)
    coef_samples = np.linspace(xy_min, xy_max, args.n_sample+2).tolist()

    all_vals = []

    columns = ["point_num", "loss"]
    for k in range(args.n_sample+2):
        coeffs_t = coef_samples[k]
        print(f'{coeffs_t}')
        linear_comb(w1, w2, coeffs_t[0], coeffs_t[1], model)
        metrics = eval(input_target_loader, model,
                       criterion, pred_fn, metric)
        if k==0:
            columns += [key for key in metrics.keys() if key!="loss"]

        values = [k, metrics["loss"]]+[v for key,v in metrics.items() if key!="loss"]

        all_vals.append(values)

        table = tabulate.tabulate([values], columns,
                                  tablefmt='simple', floatfmt='8.4f')

        print(table, flush=True)

    return all_vals, euclidean_dist

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Interpolate between model pairs.")

    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma separated list of models to interpolate between. "
        "Or a substring. All models on hf-hub having this substring will "
        "be interpolated between.",
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="bert-base-uncased",
        help="Model name to be used to load the tokenizer "
             "(default: bert-base-uncased).",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="input batch size (default: 512)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnli",
        choices=["mnli", "hans", "cola", "qqp", "paws"],
        help="dataset to evaluate on [mnli, hans, cola, qqp, paws]. (default: mnli)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "in_domain_dev", "out_domain_dev", "validation", "dev_and_test"],
        help="data split to use: [train, test](for mnli/hans) or "
            "[train, in_domain_dev, out_domain_dev, validation](for cola) or "
            "[dev_and_test, train](for paws) or [train, validation, test](qqp). "
            "For MNLI, 'test' resolves to 'validation_matched', and for HANS, "
            "'test' resolves to 'validation'. (default: test)",
    )

    parser.add_argument(
        "--num_exs",
        type=int,
        default=512,
        help="number of examples used to evaluate each model on "
              "the linear interpolation curve.(default: 512)",
    )

    parser.add_argument(
        "--n_sample",
        type=int,
        default=8,
        help="number of samples between the points we are"
              "interpolating between. (default: 8)",
    )

    parser.add_argument(
        "--save_file",
        type=str,
        required=True,
        help="Name of file where to save the interpolation values,"
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
        help="In case args.models is a substring, this tells "
        "which exact pair of models to interpolate between."
    )

    args = parser.parse_args()

    vals_dict = {}

    if args.steps is None:
        args.steps = (None, None)
    else:
        args.steps = args.steps.split(',')
        if len(args.steps)==1:
            args.steps = (args.steps[0], args.steps[0])
    
    args.models = tuple(args.models.split(','))
    if len(args.models)==1:
        args.models = get_model_pairs(args.models[0], args.steps[0])[args.job_id]

    if args.job_id is not None:
        args.save_file = ('.'.join(args.save_file.split('.')[:-1])
                          +f'_{args.job_id}.'+args.save_file.split('.')[-1])

    args.experiment_id = args.save_file.replace('/', '_')

    args.from_model_type = get_model_type(args.models[0])
    assert args.from_model_type==get_model_type(args.models[1])

    linear_interpol_vals, euclidean_dist = main(args)
    vals_dict[args.models] = (linear_interpol_vals, euclidean_dist)
    vals_dict[args.models] = (linear_interpol_vals[::-1], euclidean_dist)
    print(f"Completed interpolation from {args.models[0]} to {args.models[1]}", flush=True)


    with open(args.save_file, "wb") as f:
        pickle.dump(vals_dict, f)

    print(f"Wrote the values to {args.save_file}!", flush=True)
