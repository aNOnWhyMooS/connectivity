import argparse
import numpy as np
import pandas as pd
import seaborn as sns

from constellations.utils.load_eval_logs import get_metrics_with_sufs
from constellations.utils.load_interpols import load_logs

import constellations.theoretical.metrics as mets

import matplotlib.pyplot as plt
plt.style.use("../paper.mplstyle")

entailing_str = "_onlyEntailing"

def get_parser():
    
    parser = argparse.ArgumentParser()

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
        choices=["mod_BH_metric"]
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

def clean_model_name(model_name):
    cleaned = "0" if (model_name=="0" or model_name=="00") else model_name.lstrip("0")
    return cleaned

def get_dist_matrix(interpol_vals, ordered_models, metric_fn):
    return np.array([[metric_fn(*interpol_vals[(("0" if (model1=="0" or model1=="00") else model1.lstrip("0")), 
                                                ("0" if (model2=="0" or model2=="00") else model2.lstrip("0")))])
                        for model2 in ordered_models] 
                        for model1 in ordered_models])

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.perf_metric=None
    accs = get_accs(args)
    
    metric_fn = getattr(mets, args.metric)
    interpol_vals = load_logs(args.interpol_log_dir)
    suf_ordered_models = list(accs.keys())
    dist_matrix = get_dist_matrix(interpol_vals, suf_ordered_models, metric_fn)
    #delta=1.0
    #dist_matrix = np.exp(-dist_matrix**2 / (2. * delta ** 2))
    Nr, Dr = 0, 0
    ks = np.sum(dist_matrix, axis=-1)
    for i in range(len(dist_matrix)):
        for j in range(len(dist_matrix)):
            for k in range(len(dist_matrix)):
                Nr+=dist_matrix[i][j]*dist_matrix[j][k]*dist_matrix[k][i]
        Dr+=ks[i]*(ks[i]-1)
    print("Clustering Coefficient:", Nr/Dr, Nr, Dr)

    suf_ordered_models = [clean_model_name(name) for name in suf_ordered_models]

    total_triplets = 0
    exact_triangle_ineq = 0
    triangle_ineq_diff = []
    percent_triangle_ineq_diff = []

    for model1 in suf_ordered_models:
        for model2 in suf_ordered_models:
            for model3 in suf_ordered_models:
                d12 = metric_fn(*interpol_vals[(model1, model2)])
                d23 = metric_fn(*interpol_vals[(model2, model3)])
                d13 = metric_fn(*interpol_vals[(model1, model3)])
                total_triplets+=1
                if d12+d23+1e-10>=d13:
                    exact_triangle_ineq+=1
                else:
                    triangle_ineq_diff.append(d13-(d12+d23))
                    if d12+d23==0:
                        print(model1, model2, model3, d12, d23, d13, d12+d23)
                    percent_triangle_ineq_diff.append(100*(d13-(d12+d23))/(d12+d23+1e-10))
    print("Total triangle inequality violation:", sum(triangle_ineq_diff))
    print("Mean triangle inequality violation:", sum(triangle_ineq_diff)/(total_triplets-exact_triangle_ineq))
    print("Exact inequality satisfied for:", exact_triangle_ineq)
    print("Total triplets:", total_triplets)
    print("Mean percentage difference:", sum(percent_triangle_ineq_diff)/(total_triplets-exact_triangle_ineq))
    print(len(percent_triangle_ineq_diff), len(triangle_ineq_diff))
    df = pd.DataFrame({"percent_violations": percent_triangle_ineq_diff})
    sns.histplot(df, x="percent_violations", kde=True,)
    plt.show()