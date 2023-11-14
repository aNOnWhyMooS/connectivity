from typing import List, OrderedDict, Tuple
import sys
import argparse
import tabulate
import evaluate

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from constellations.dataloaders.loader import get_loader
from constellations.simplexes.orig_utils import eval_model
from constellations.model_loaders.load_model import get_sequence_classification_model
from constellations.model_loaders.modelling_utils import get_pred_fn, get_criterion_fn, get_logits_converter, linear_comb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def euclidean_dist(x: OrderedDict, y: OrderedDict) -> float:
    """Calculates Euclidean distance between two ordered dictionaries corresponding
    to model state dicts. Assumes Buffer values are same for both dicts."""
    return torch.sqrt(sum([torch.sum((v1-v2)*(v1-v2))
                           for (k1, v1), (k2, v2) in
                           zip(x.items(), y.items())])).item()

def get_points_on_bent_line(dists: List[float],
                            step_size: float,) -> List[List[Tuple[float, float]]]:
    """Args:
        dists: List of distances between consecutive points on the line.
        step_size: Step size to take between points on the line.
    Returns:
        List of lists of tuples of form (coeff_1, coeff_2) where:
            1. Each inner list corresponds to a segment of the bent line between two
            consecutive vertices.
            2. coeff1, coeff2 are scaling coefficients for the two end points of the
            current segment, with coeff1+coeff2=1 and coeff1>0, coeff2>0.
            3. The outer list is a list over all segments.
    """
    intermediate_points = []
    prev_dist = 0.
    for dist in dists:
        points_on_segment = [(1.,0.)]
        while True:
            left_point_wt = points_on_segment[-1][0]-(step_size-prev_dist)/dist
            right_point_wt = points_on_segment[-1][1]+(step_size-prev_dist)/dist
            if left_point_wt<0 or right_point_wt>1:
                prev_dist = dist-step_size*int(dist/step_size)
                break
            points_on_segment.append((left_point_wt,right_point_wt))
            prev_dist = 0.
        intermediate_points.append(points_on_segment)
    intermediate_points[-1].append((0.0, 1.0))
    return intermediate_points

def get_intermediate_points(args,
                            points: List[OrderedDict]) -> List[List[Tuple[float, float]]]:
    """Returns the segment-wise scaling weights for each segment in the bent line
    specified by points list."""
    dists = [euclidean_dist(points[i], points[i+1])
             for i in range(len(points)-1)]

    print(f"Distances between points on bent line: {dists}", flush=True)

    min_dist = min(dists)
    step_size = min_dist/args.min_pts_bw_models
    intermediate_points = get_points_on_bent_line(dists, step_size)

    print(f"Considering {sum([len(elem) for elem in intermediate_points])} \
            points on the bent line.", flush=True)

    return intermediate_points

def main(args):
    mnli_label_dict = {"contradiction": 0, "entailment": 1, "neutral": 2}
    hans_label_dict = {"entailment": 0, "non-entailment": 1}

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if args.dataset=="hans" and args.all_data:
        input_target_loader = get_loader(args, tokenizer, mnli_label_dict,
                                         heuristic_wise=["lexical_overlap",
                                                         "constituent", "subsequence"],
                                         onlyNonEntailing=False)
    else:
        input_target_loader = get_loader(args, tokenizer, mnli_label_dict)

    ce_loss = nn.CrossEntropyLoss()

    mnli_logits_to_hans = get_logits_converter(mnli_label_dict, hans_label_dict)

    logit_converter = mnli_logits_to_hans if args.dataset=="hans" else None

    criterion = get_criterion_fn(ce_loss, logit_converter)
    pred_fn   = get_pred_fn(pred_type="argmax",
                            logit_converter_fn = logit_converter)

    if args.dataset in ["mnli", "hans"]:
        metric = evaluate.load("accuracy", experiment_id=args.experiment_id)
    elif args.dataset in ["qqp", "paws"]:
        metric = evaluate.load("glue", "qqp", experiment_id=args.experiment_id)

    points = [get_sequence_classification_model(args.base_models_prefix + args.models[i]).state_dict()
              for i in range(len(args.models))]

    model = get_sequence_classification_model(args.base_models_prefix + args.models[0])

    columns = ["point_num", "loss", "accuracy"]

    coeffs = get_intermediate_points(args, points)
    k = 0
    for i, coeffs_segment in enumerate(coeffs):
        print(f"Interpolating in segment from {i} to {i+1}", flush=True)
        for coeffs in coeffs_segment:
            print("Coefficients:", coeffs, flush=True)

            linear_comb(points[i], points[i+1], coeffs[0], coeffs[1], model)
            metrics = eval_model(input_target_loader, model,
                                 criterion, pred_fn, metric)
            values = [k, metrics['loss'], metrics['accuracy'],]
            table = tabulate.tabulate([values], columns,
                                      tablefmt='simple', floatfmt='8.8f')
            k+=1

            print(table, flush=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="bert simplex")

    parser.add_argument(
        "--base_models_prefix",
        type=str,
        help="Common prefix of models to be loaded",
    )

    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Model name to be used to load the tokenizer \
             for the models.",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Suffixes to add to base model prefix, to get \
            the models which compose the line with n-bends.",
    )

    parser.add_argument(
        "--min_pts_bw_models",
        type=int,
        default=20,
        help="Minimum number of points to sample from between two consecutive models.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="input batch size (default: 50)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnli",
        choices=["mnli", "hans", "qqp", "paws"],
        help="dataset [mnli, hans, qqp, paws]",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "validation", "dev_and_test"],
        help="dataset [train, test]",
    )

    parser.add_argument(
        "--num_exs",
        type=int,
        default=512,
        help="number of examples used to evaluate each model on \
              the linear interpolation curve.",
    )

    parser.add_argument(
        "--all_data",
        action="store_true",
        default=False,
        help="whether to use all data for evaluation on HANS dataset. By default, \
            only nonEntailing samples of lexical_overlap heuristic are used.",
    )

    parser.add_argument(
        "--experiment_id",
        type=str,
        default='',
        required=False,
        help="Experiment ID, required if multiple experiments are sharing same file system.",
    )

    parser.add_argument(
        "--paws_data_dir",
        type=str,
        default="../paws_final",
        help="Data directory having final paws-qqp files(default: ../paws_final)."
    )

    args = parser.parse_args()

    main(args)
