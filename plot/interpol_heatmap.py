import argparse
import numpy as np
import itertools
import warnings

from constellations.plot_utils import plot_heatmap, plot_lineplot

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
        "--order_by",
        choices=["perf", "cluster", "seed"],
        required=True,
    )
    
    parser.add_argument(
        "--perf_metric",
        choices=([elem1+elem2 for (elem1, elem2) in 
                 itertools.product(["subsequence", "lexical_overlap", "constituent"],
                                    ["_onlyNonEntailing", entailing_str])]+["in_domain_dev", "out_domain_dev"]),
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
        help="The prefix for .json model evaluation files."
    )

    parser.add_argument(
        "--eval_mods_suffixes",
        nargs="+",
        required=True,
        help="The suffixes for .json model evaluation files.\
            Don't include .json in suffix. Must match the model ids\
            in interpolation files."
    )

    parser.add_argument(
        "--metric",
        default="mod_BH_metric",
        choices=["mod_BH_metric", "BaH_metric", "AreaUnderCurve"]
    )

    parser.add_argument(
        "--ticks",
        default="models",
        choices=["accs", "models"],
        help="What ticklabels to show on the axes."
    )

    parser.add_argument(
        "--emb_acc_corr",
        action="store_true",
        help="If provided, spectral embedding(2 clusters) and accuracy's\
        pearson correlation coeff. will be calculated and a line plot \
            will be shown."
    )

    parser.add_argument(
        "--eval_metric",
        type=str,
        help="Metric to use to rank models.(default: accuracy)",
        default="accuracy",
    )
    return parser


def get_accs(args):
    
    if "cola" in args.eval_mods_prefix:
        data_type="cola"
    elif "qqp" in args.eval_mods_prefix or "paws" in args.eval_mods_prefix:
        data_type="qqp"
    else:
        data_type="nli"
    
    print(f"Inferred data type: {data_type} for the provided\
        file prefix {args.eval_mods_prefix}")
    
    if "_" in str(args.perf_metric):
        if "Entailing" in str(args.perf_metric):
            heuristic = "_".join(args.perf_metric.split("_")[:-1])
            entailing = False
            accs = {k: v[args.eval_metric] for k, v in get_metrics_with_sufs(args.eval_mods_prefix, 
                                                              args.eval_mods_suffixes, 
                                                              heuristic, entailing).items()}
        else:
            accs = {k: v[args.eval_metric] for k, v in get_metrics_with_sufs(args.eval_mods_prefix, 
                                                              args.eval_mods_suffixes,
                                                              split=args.perf_metric,
                                                              data_type=data_type).items()}
    else:
        accs = {k: v[args.eval_metric] for k, v in get_metrics_with_sufs(args.eval_mods_prefix, 
                                                          args.eval_mods_suffixes,
                                                          data_type=data_type).items()}
    return accs

def get_dist_matrix(interpol_vals, ordered_models, metric_fn):
    return np.array([[metric_fn(*interpol_vals[(("0" if (model1=="0" or model1=="00") else model1.lstrip("0")), 
                                                ("0" if (model2=="0" or model2=="00") else model2.lstrip("0")))])
                        for model2 in ordered_models] 
                        for model1 in ordered_models])


if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    accs = get_accs(args)
    
    assert list(accs.keys())==args.eval_mods_suffixes
    
    if args.ticks=="accs":
        assert args.order_by=="perf"
    
    metric_fn = getattr(mets, args.metric)
    interpol_vals = load_logs(args.interpol_log_dir)
    suf_ordered_models = list(accs.keys())
    dist_matrix = get_dist_matrix(interpol_vals, suf_ordered_models, metric_fn)
    
    for i in range(len(dist_matrix)):
        if dist_matrix[i][i]!=0:
            model_name = "0" if (suf_ordered_models[i]=="0" or suf_ordered_models[i]=="00") else suf_ordered_models[i].lstrip("0")
            warnings.warn(f"Non zero entry on diagonal with metric {args.metric}\
                at location {i} with interpol vals: {interpol_vals[(model_name, model_name)]}. Got value {dist_matrix[i][i]} instead.")
    
    clusters = get_clusters(dist_matrix, args.n_clusters)
        
    clusters = [[suf_ordered_models[idx] for idx in cluster] 
                                         for cluster in clusters]
    
    print("Clusters found:", clusters)        

    if args.order_by=="cluster":
    
        ordered_models = list(itertools.chain.from_iterable(clusters))
    
    elif args.order_by=="perf":
        ordered_models = sorted(suf_ordered_models, key=lambda e: accs[e])
    else:
        ordered_models = suf_ordered_models
    
    ordered_model_indices = [suf_ordered_models.index(model) for model in ordered_models]
    
    if args.ticks=="accs":
        ticks = [f"{accs[model]:.3f}" for model in ordered_models]
        print(ticks)
    else:
        ticks = ordered_models
    
    save_file = (args.interpol_log_dir.replace("/", "_").replace(".", "")+"--"
                +args.eval_mods_prefix.replace("/", "_")+"--"
                +"order_by="+args.order_by+"--"
                +("" if args.perf_metric is None 
                     else "perf_metric="+args.perf_metric)+"--"
                +"metric="+args.metric+"--").replace("..", "").replace("--", "-")
    
    save_file = ''.join(elem.title() for elem in save_file.split("_"))
    
    heatmap_save_file = ("interpolate="+args.interpol_log_dir.strip("/").split("/")[-1]
                         +"_perf_metric="+str(args.perf_metric)+".pdf")
    
    lineplot_save_file = ("interpolate="+args.interpol_log_dir.strip("/").split("/")[-1]
                          +"_perf_metric="+str(args.perf_metric)+".pdf")
    
    if args.perf_metric is not None:
        ylabel = args.perf_metric
    elif "mnli" in args.eval_mods_prefix:
        ylabel="MNLI"
    elif "hans" in args.eval_mods_prefix:
        ylabel="HANS"
    elif "paws" in args.eval_mods_prefix:
        ylabel="PAWS"
    elif "qqp" in args.eval_mods_prefix:
        ylabel="QQP"
    elif "cola" in args.eval_mods_prefix:
        ylabel="CoLA"
    else:
        warnings.warn(f"Unable to figure out perf metric for\
            line-plot from eval_mods_prefix {args.eval_mods_prefix}")
    
    plot_heatmap(dist_matrix, ordered_model_indices, ticks, heatmap_save_file, args.ticks, ylabel, args.eval_metric)
    
    plot_lineplot(dist_matrix, accs, clusters, lineplot_save_file, ylabel, eval_metric=args.eval_metric)
