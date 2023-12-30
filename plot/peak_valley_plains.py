import argparse
import pandas as pd
import numpy as np
import itertools

from constellations.plot_utils import plot_plain_valley_peaks

from constellations.utils.load_eval_logs import get_metrics_with_sufs
from constellations.utils.load_interpols import load_logs


rename = {
    "peaks": "gen.",
    "valleys": "heur.",
    "plains": "middle",
}


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--perf_metric",
        choices=(
            [
                elem1 + elem2
                for (elem1, elem2) in itertools.product(
                    ["subsequence", "lexical_overlap", "constituent"],
                    ["_onlyNonEntailing"],
                )
            ]
            + ["in_domain_dev", "out_domain_dev"]
        ),
        help="By default all samples in the .json files in eval_log_dir will be used.",
    )

    parser.add_argument(
        "--interpol_log_dirs",
        nargs="+",
        required=False,
        help="The directory to load interpol logs from.",
    )

    parser.add_argument(
        "--interpol_datasets",
        nargs="+",
        required=False,
        help="The names of datasets in which interpolation\
            has been done, and provided in --interpol_log_dir.",
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
        "--num_models_per_class",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--only_print_models",
        action="store_true",
        help="If provided, only peak, plain and valley\
            models will be printed. To be used if no\
            interpolation data available yet.",
    )

    parser.add_argument(
        "--remove_plains",
        action="store_true",
        help="Don't include plain models in the plots.",
    )

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
            [metric_fn(*interpol_vals[(model1, model2)]) for model2 in ordered_models]
            for model1 in ordered_models
        ]
    )


def get_peak_valley_plain(args, accs):
    models_sorted_by_acc = sorted(list(accs.keys()), key=lambda e: accs[e])
    valleys = models_sorted_by_acc[: args.num_models_per_class]
    peaks = models_sorted_by_acc[-args.num_models_per_class :]
    mid_point = len(accs) // 2
    width = args.num_models_per_class // 2
    if args.num_models_per_class % 2 == 0:
        plains = models_sorted_by_acc[mid_point - width : mid_point + width]
    else:
        plains = models_sorted_by_acc[mid_point - width : mid_point + width + 1]
    return peaks, valleys, plains


def get_dataframe(peaks, valleys, plains, interpol_losses, interpol_accs, dataset_name):
    inter_dict = {
        "position": [],
        "losses": [],
        "type": [],
        "accs": [],
        "dataset_names": [],
    }

    def append_data(
        lis1,
        lis2,
        lis1_type,
        lis2_type,
    ):
        same = lis1 == lis2
        for i, model1 in enumerate(lis1):
            for j, model2 in enumerate(lis2):
                if i == j and same:
                    continue
                model1, model2 = (
                    ("0" if (model1 == "0" or model1 == "00") else model1.lstrip("0")),
                    ("0" if (model2 == "0" or model2 == "00") else model2.lstrip("0")),
                )
                _, losses = interpol_losses[(model1, model2)]
                _, accs = interpol_accs[(model1, model2)]
                assert len(losses) == len(accs)
                inter_dict["losses"] += losses
                inter_dict["accs"] += accs
                inter_dict["type"] += [
                    rename[lis1_type] + "-to-" + rename[lis2_type]
                ] * len(losses)
                inter_dict["position"] += list(range(len(losses)))
                inter_dict["dataset_names"] += [dataset_name] * len(losses)

    if args.remove_plains:
        models = {"peaks": peaks, "valleys": valleys}
    else:
        models = {"peaks": peaks, "valleys": valleys, "plains": plains}

    for i, (lis1_type, lis1) in enumerate(models.items()):
        for j, (lis2_type, lis2) in enumerate(models.items()):
            if i > j:
                continue
            append_data(lis1, lis2, lis1_type, lis2_type)

    return pd.DataFrame(inter_dict)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    assert args.interpol_log_dirs is None or len(args.interpol_datasets) == len(
        args.interpol_log_dirs
    )

    accs = get_accs(args)

    assert list(accs.keys()) == args.eval_mods_suffixes
    peaks, valleys, plains = get_peak_valley_plain(args, accs)

    print("Peak models & their accs:", peaks, [accs[i] for i in peaks], flush=True)
    print(
        "Valley models & their accs:", valleys, [accs[i] for i in valleys], flush=True
    )
    print("Plain models & their accs:", plains, [accs[i] for i in plains], flush=True)

    if args.only_print_models:
        exit(0)

    dfs = []
    for interpol_dir, dataset_name in zip(
        args.interpol_log_dirs, args.interpol_datasets
    ):
        interpol_losses = load_logs(interpol_dir)
        interpol_accs = load_logs(interpol_dir, metric="acc")
        df = get_dataframe(
            peaks, valleys, plains, interpol_losses, interpol_accs, dataset_name
        )
        dfs.append(df)
    df = pd.concat(dfs)

    save_file = (
        (
            "_".join(args.interpol_datasets)
            + "--"
            + args.eval_mods_prefix.replace("/", "_")
            + "--"
            + ("" if args.perf_metric is None else "perf_metric=" + args.perf_metric)
        )
        .replace("..", "")
        .replace("--", "-")
    )

    save_file = "".join(elem.title() for elem in save_file.split("_"))
    df = df.rename(columns={"dataset_names": "data"})
    plot_plain_valley_peaks(
        df, save_file + "_losses.pdf", "type", "position", "losses", "data"
    )
    plot_plain_valley_peaks(
        df, save_file + "_accs.pdf", "type", "position", "accs", "data"
    )
