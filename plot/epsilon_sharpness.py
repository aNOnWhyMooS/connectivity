import argparse
import numpy as np
import itertools
import re

from constellations.plot_utils import lineplot

from constellations.utils.load_eval_logs import get_metrics_with_sufs

import matplotlib.pyplot as plt

plt.style.use("../paper.mplstyle")

entailing_str = "_onlyEntailing"


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--perf_metric",
        choices=(
            [
                elem1 + elem2
                for (elem1, elem2) in itertools.product(
                    ["subsequence", "lexical_overlap", "constituent"],
                    ["_onlyNonEntailing", entailing_str],
                )
            ]
        ),
        help="By default all samples in the .json files in eval_log_dir will be used.",
    )

    parser.add_argument(
        "--eval_mods_prefix",
        required=True,
        help="The prefix for .json model evaluation files.",
    )

    parser.add_argument(
        "--eval_mods_suffixes",
        nargs="+",
        required=True,
        help="The suffixes for .json model evaluation files.\
            Don't include .json in suffix. Must match the model ids\
            in interpolation files.",
    )

    parser.add_argument(
        "--n_samples",
        default=8192,
        choices=[8192, 2048],
        help="Choose how many samples were used to evaluate epsilon-sharpness.",
    )

    return parser


def get_accs(args):
    if "_" in str(args.perf_metric):
        heuristic = "_".join(args.perf_metric.split("_")[:-1])
        entailing = entailing_str == args.perf_metric.split("_")[-1]
        accs = {
            k: v[1]
            for k, v in get_metrics_with_sufs(
                args.eval_mods_prefix, args.eval_mods_suffixes, heuristic, entailing
            ).items()
        }
    else:
        accs = {
            k: v[1]
            for k, v in get_metrics_with_sufs(
                args.eval_mods_prefix, args.eval_mods_suffixes
            ).items()
        }
    return accs


def get_dist_matrix(interpol_vals, ordered_models, metric_fn):
    return np.array(
        [
            [
                metric_fn(
                    *interpol_vals[
                        (
                            (
                                "0"
                                if (model1 == "0" or model1 == "00")
                                else model1.lstrip("0")
                            ),
                            (
                                "0"
                                if (model2 == "0" or model2 == "00")
                                else model2.lstrip("0")
                            ),
                        )
                    ]
                )
                for model2 in ordered_models
            ]
            for model1 in ordered_models
        ]
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    log_file = f"../logs/NLI/feather_berts/flatness_measure/mnli_flatness_measure_{args.n_samples}_samples.log"
    with open(log_file) as f:
        lines = f.readlines()

    flatness = {}
    for line in lines:
        match_obj = re.match(
            r"ɛ-sharpness of ../../../100_berts/bert_(\d\d) is (\d+\.\d+)", line
        )
        if match_obj is not None:
            model_no = int(match_obj.group(1))
            score = float(match_obj.group(2))
            flatness[model_no] = score

    inps = list(flatness.values())
    print(flatness.keys())
    accs = get_accs(args)
    outs = [
        accs[("0" if (k < 10 and "feather" in args.eval_mods_prefix) else "") + str(k)]
        for k in flatness
    ]

    save_file = "epsilon_sharpness_" + args.perf_metric + ".pdf"
    lineplot(
        np.array(inps),
        np.array(outs),
        [list(range(len(inps)))],
        save_file,
        ylabel=args.perf_metric,
        xlabel=r"$\epsilon$-sharpness",
    )
