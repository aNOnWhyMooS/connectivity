import pickle
import collections
import argparse
from typing import OrderedDict

import tabulate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_metric
from transformers import AutoTokenizer

from constellations.model_loaders.modelling_utils import get_criterion_fn, get_logits_converter, get_pred_fn
from constellations.model_loaders.load_model import get_sequence_classification_model
from constellations.dataloaders.loader import get_loader
from constellations.utils.eval_utils import eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def linear_comb(w1: OrderedDict[str, torch.Tensor], 
                w2: OrderedDict[str, torch.Tensor], 
                coeff1: float, coeff2: float,
                model: nn.Module) -> None:
    """Linearly combines weights w1 and w2 as coeff1*w1 and coeff2*w2 and loads
    into provided model.
    Args:
        w1:     State dict of first model.
        w2:     State dict of second model.
        coeff1: Coefficient for scaling weights in w1.
        coeff2: Coefficient for scaling weights in w2.
        model:   The model in which to load the linear combination of w1 and w2.
    """
    new_state_dict = collections.OrderedDict()
    buffers = [name for (name, _) in model.named_buffers()]

    for (k1, v1), (k2, v2) in zip(w1.items(), w2.items()):
        if k1!=k2:
            raise ValueError(f"Mis-matched keys {k1} and {k2} encountered while \
                               forming linear combination of weights.")
        if k1 not in buffers:
            new_state_dict[k1] = coeff1*v1+coeff2*v2
        else:
            new_state_dict[k1] = v1
    model.load_state_dict(new_state_dict)
    model.to(device)

def main(args):
    is_feather_bert = (("feather" in args.base_models_prefix+args.indices[0])
                    or ("feather" in args.base_models_prefix+args.indices[1]))

    if is_feather_bert:
        assert args.num_steps==36813 or args.num_steps is None
        args.num_steps=None
        mnli_label_dict = {"contradiction": 0, "entailment": 1, "neutral": 2}
    else:        
        mnli_label_dict = {"contradiction": 2, "entailment": 0, "neutral": 1}
    
    hans_label_dict = {"entailment": 0, "non-entailment": 1}

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except:
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
    
    w1 = get_sequence_classification_model(
            args.base_models_prefix + args.indices[0], from_flax=not is_feather_bert,
            num_steps=args.num_steps, local_dir=(None if args.local_dir_prefix is None
                                                      else args.local_dir_prefix + args.indices[0]),
        ).state_dict()
    
    model = get_sequence_classification_model(
            args.base_models_prefix + args.indices[0], from_flax=not is_feather_bert,
            num_steps=args.num_steps, local_dir=(None if args.local_dir_prefix is None
                                                      else args.local_dir_prefix + args.indices[0]),
        )
    
    w2 = get_sequence_classification_model(
            args.base_models_prefix  + args.indices[1], from_flax=not is_feather_bert,
            num_steps=args.num_steps, local_dir=(None if args.local_dir_prefix is None
                                                      else args.local_dir_prefix + args.indices[1]),
        ).state_dict()
    
    euclidean_dist = torch.sqrt(sum([torch.sum((v1-v2)*(v1-v2)) for (_, v1), (_, v2) in zip(w1.items(), w2.items())])).item()
    print(f"Euclidean distance between {args.indices[0]} and {args.indices[1]}: {euclidean_dist}.", flush=True)
    print(f'Interpolating between {args.indices[0]} and {args.indices[1]} on {args.dataset}', flush=True)

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
        "--base_models_prefix",
        type=str,
        required=True,
        help="Common prefix of models to be loaded(e.g. 'connectivity/bert_ft_qqp-')",
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="bert-base-uncased",
        help="Model name to be used to load the tokenizer \
              for the models, if tokenizer not found in\
              args.base_models_prefix+args.models[i]. (default: bert-base-uncased)",
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
        help="data split to use: [train, test](for mnli/hans) or \
            [train, in_domain_dev, out_domain_dev, validation](for cola) or\
            [dev_and_test, train](for paws) or [train, validation, test](qqp).\
            For MNLI, 'test' resolves to 'validation_matched', and for HANS,\
            'test' resolves to 'validation'. (default: test)",
    )

    parser.add_argument(
        "--num_exs",
        type=int,
        default=512,
        help="number of examples used to evaluate each model on \
              the linear interpolation curve.(default: 512)",
    )

    parser.add_argument(
        "--n_sample",
        type=int,
        default=8,
        help="number of samples between the points we are\
              interpolating between. (default: 8)",
    )
    
    parser.add_argument(
        "--save_file",
        type=str,
        required=True,
        help="Name of file where to save the interpolation values,\
        using pickle."
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        help="Load model on the remote at these number of steps, \
            for evaluation. A commit with this number of '\d+ steps' in its\
            commit message, must be present on the remote. By default, latest\
            model will be loaded.",
    )

    parser.add_argument(
        "--local_dir_prefix",
        type=str,
        help="Local directory prefix to fetch commits from(if repository is\
            already cloned on local machine). For use with num_steps.\
            If not specified commit info will be fetched\
            into dummy directory from huggingface.co/",
    )

    parser.add_argument(
        "--experiment_id",
        type=str,
        help="Experiment id to use for using HF metrics in\
              distributed storage systems with multiple processes\
              running in parallel.",
    )
    
    parser.add_argument(
        "--suffix_pairs",
        type=str,
        nargs="+",
        help="pairs of suffixes(comma separated) to add to \
              base_models_prefix to get models to interpolate between.",
        required=True,
    )
    
    args = parser.parse_args()

    vals_dict = {}
    for suffix_pair in args.suffix_pairs:
        model1, model2 = suffix_pair.split(",")
        args.indices = (model1, model2)
        
        try:
            linear_interpol_vals, euclidean_dist = main(args)
        except ValueError as e:
            if "Unable to find any commit" in str(e):
                print(e, "for model pair:", (model1, model2), flush=True)
                break
            else:
                raise e
        
        vals_dict[(model1,model2)] = (linear_interpol_vals, euclidean_dist)
        vals_dict[(model2,model1)] = (linear_interpol_vals[::-1], euclidean_dist)
        print(f"Completed interpolation from {model1} to {model2}", flush=True)

    
    with open(args.save_file, "wb") as f:
        pickle.dump(vals_dict, f)
    
    print(f"Wrote the values to {args.save_file}!", flush=True)
