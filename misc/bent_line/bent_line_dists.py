from typing import List, OrderedDict, Tuple
import collections
import sys
import argparse

import torch
import torch.nn as nn

import constellations.legacy.common_utils as cu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_comb(
    w1: OrderedDict[str, torch.Tensor],
    w2: OrderedDict[str, torch.Tensor],
    coeff1: float,
    coeff2: float,
    model: nn.Module,
) -> None:
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
        if k1 != k2:
            raise ValueError(
                f"Mis-matched keys {k1} and {k2} encountered while \
                               forming linear combination of weights."
            )
        if k1 not in buffers:
            new_state_dict[k1] = coeff1 * v1 + coeff2 * v2
        else:
            new_state_dict[k1] = v1
    model.load_state_dict(new_state_dict)
    model.to(device)


def euclidean_dist(x: OrderedDict, y: OrderedDict) -> float:
    """Calculates Euclidean distance between two ordered dictionaries corresponding
    to model state dicts. Assumes Buffer values are same for both dicts."""
    return torch.sqrt(
        sum(
            [
                torch.sum((v1 - v2) * (v1 - v2))
                for (k1, v1), (k2, v2) in zip(x.items(), y.items())
            ]
        )
    ).item()


def get_points_on_bent_line(
    dists: List[float],
    step_size: float,
) -> List[List[Tuple[float, float]]]:
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
    prev_dist = 0.0
    for dist in dists:
        points_on_segment = [(1.0, 0.0)]
        while True:
            left_point_wt = points_on_segment[-1][0] - (step_size - prev_dist) / dist
            right_point_wt = points_on_segment[-1][1] + (step_size - prev_dist) / dist
            if left_point_wt < 0 or right_point_wt > 1:
                prev_dist = dist - step_size * int(dist / step_size)
                break
            points_on_segment.append((left_point_wt, right_point_wt))
            prev_dist = 0.0
        intermediate_points.append(points_on_segment)
    return intermediate_points


def get_intermediate_points(
    args, points: List[OrderedDict]
) -> List[List[Tuple[float, float]]]:
    """Returns the segment-wise scaling weights for each segment in the bent line
    specified by points list."""
    dists = [euclidean_dist(points[i], points[i + 1]) for i in range(len(points) - 1)]

    print(f"Distances between points on bent line: {dists}", flush=True)

    min_dist = min(dists)
    step_size = min_dist / args.min_pts_bw_models
    intermediate_points = get_points_on_bent_line(dists, step_size)

    print(
        f"Considering {sum([len(elem) for elem in intermediate_points])} \
            points on the bent line.",
        flush=True,
    )

    return intermediate_points


def main(args):
    points = [
        cu.get_sequence_classification_model(
            args.base_models_prefix + args.models[i]
        ).state_dict()
        for i in range(len(args.models))
    ]

    first_point = collections.OrderedDict(
        {k: v.to(device) for k, v in points[0].items()}
    )
    last_point = collections.OrderedDict(
        {k: v.to(device) for k, v in points[-1].items()}
    )

    model = cu.get_sequence_classification_model(
        args.base_models_prefix + args.models[0]
    )

    coeffs = get_intermediate_points(args, points)

    k = 0
    for i, coeffs_segment in enumerate(coeffs):
        print(f"Interpolating in segment from {i} to {i+1}", flush=True)
        for coeffs in coeffs_segment:
            print("Coefficients:", coeffs, flush=True)

            linear_comb(points[i], points[i + 1], coeffs[0], coeffs[1], model)

            dist_from_first_model = euclidean_dist(model.state_dict(), first_point)
            dist_from_last_model = euclidean_dist(model.state_dict(), last_point)

            print(
                f"Distance of {k}-th model from first model: {dist_from_first_model}",
                flush=True,
            )
            print(
                f"Distance of {k}-th model from last model: {dist_from_last_model}",
                flush=True,
            )
            k += 1


if __name__ == "__main__":
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
        default=128,
        help="input batch size (default: 50)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnli",
        choices=["mnli", "hans"],
        help="dataset [mnli, hans]",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="dataset [train, test]",
    )

    parser.add_argument(
        "--num_exs",
        type=int,
        default=1000,
        help="number of examples used to evaluate each model on \
              the linear interpolation curve.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="acc",
        choices=["acc", "ECE"],
        help="metric to be used for evaluation, use acc or ECE",
    )

    parser.add_argument(
        "--all_data",
        action="store_true",
        default=False,
        help="whether to use all data for evaluation on HANS dataset. By default, \
            only nonEntailing samples of lexical_overlap heuristic are used.",
    )

    args = parser.parse_args()

    main(args)
