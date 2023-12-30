import argparse
import numpy as np
import itertools
import warnings

from constellations.utils.load_eval_logs import get_metrics_with_sufs
from constellations.utils.load_interpols import load_logs

import constellations.theoretical.metrics as mets
from constellations.theoretical.clustering import get_clusters

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
            + ["in_domain_dev", "out_domain_dev"]
        ),
        help="By default all samples in the .json files in eval_log_dir will be used.",
    )

    parser.add_argument(
        "--n_clusters",
        type=int,
        required=False,
        default=2,
    )

    parser.add_argument(
        "--interpol_log_dir",
        required=True,
        help="The directory to load interpol logs from.",
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

    parser.add_argument("--metric", default="mod_BH_metric", choices=["mod_BH_metric"])

    parser.add_argument(
        "--eval_metric",
        type=str,
        help="Metric to use to rank models.",
        default="accuracy",
    )
    return parser


def get_accs(args):
    if "cola" in args.eval_mods_prefix:
        data_type = "cola"
    elif "qqp" in args.eval_mods_prefix or "paws" in args.eval_mods_prefix:
        data_type = "qqp"
    else:
        data_type = "nli"

    print(
        f"Inferred data type: {data_type} for the provided\
        file prefix {args.eval_mods_prefix}"
    )

    if "_" in str(args.perf_metric):
        if "Entailing" in str(args.perf_metric):
            heuristic = "_".join(args.perf_metric.split("_")[:-1])
            entailing = False
            accs = {
                k: v[args.eval_metric]
                for k, v in get_metrics_with_sufs(
                    args.eval_mods_prefix, args.eval_mods_suffixes, heuristic, entailing
                ).items()
            }
        else:
            accs = {
                k: v[args.eval_metric]
                for k, v in get_metrics_with_sufs(
                    args.eval_mods_prefix,
                    args.eval_mods_suffixes,
                    split=args.perf_metric,
                    data_type=data_type,
                ).items()
            }
    else:
        accs = {
            k: v[args.eval_metric]
            for k, v in get_metrics_with_sufs(
                args.eval_mods_prefix, args.eval_mods_suffixes, data_type=data_type
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
    accs = get_accs(args)

    assert list(accs.keys()) == args.eval_mods_suffixes

    metric_fn = getattr(mets, args.metric)
    interpol_vals = load_logs(args.interpol_log_dir)
    suf_ordered_models = list(accs.keys())
    dist_matrix = get_dist_matrix(interpol_vals, suf_ordered_models, metric_fn)

    for i in range(len(dist_matrix)):
        if dist_matrix[i][i] != 0:
            warnings.warn(f"Non zero entry on diagonal with metric {args.metric}")

    clusters = get_clusters(dist_matrix, args.n_clusters)

    clusters = [[suf_ordered_models[idx] for idx in cluster] for cluster in clusters]
    print("Clusters:", clusters)
    print("Total mean:", np.mean(list(accs.values())))
    print("Total std:", np.std(list(accs.values())))
    print("Total max:", np.max(list(accs.values())))
    print("Total min:", np.min(list(accs.values())))

    for i, cluster in enumerate(clusters):
        print(f"Cluster-{i} of length:", len(cluster))
        cluster_accs = [accs[e] for e in cluster]
        print("\tMaximum:", max(cluster_accs))
        print("\tMean:", np.mean(cluster_accs))
        print("\tStandard deviation:", np.std(cluster_accs))
