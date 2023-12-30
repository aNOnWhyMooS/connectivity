from typing import List

import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter

from scipy.stats import pearsonr
import seaborn as sns
import numpy as np
import pandas as pd

from .theoretical.clustering import get_sc_centroid_dists


def shorten(name):
    mapping = {
        "lexical_overlap_onlyNonEntailing": "HANS-LO",
        "lexical_overlap_onlyEntailing": "LO_ent",
        "constituent_onlyNonEntailing": "HANS-constituent",
        "constituent_onlyEntailing": "const_ent",
        "subsequence_onlyNonEntailing": "HANS-subsequence",
        "subsequence_onlyEntailing": "subseq_ent",
        "mnli": "MNLI",
        "None": "MNLI",
        "hans": "HANS",
        "matthews_correlation": "MCC",
        "accuracy": "Acc.",
        "out_domain_dev": "CoLA-OOD",
        "in_domain_dev": "CoLA-ID",
    }
    try:
        return mapping[name]
    except KeyError:
        return name


def heatmap(
    heats: np.ndarray,
    xticklabels,
    yticklabels,
    save_file,
    tick_typs="models",
    ylabel="MNLI",
    eval_metric="accuracy",
):
    plt.figure(figsize=(10, 8), dpi=100)

    ax = sns.heatmap(
        heats,
        yticklabels=yticklabels,
        xticklabels=xticklabels,
    )

    ticks = np.linspace(
        0, len(heats) - 1, (8 if tick_typs == "accs" else 25), dtype=np.int
    )
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([xticklabels[i] for i in ticks])
    ax.set_yticklabels([yticklabels[i] for i in ticks])

    if tick_typs == "accs":
        ax.set_xlabel(f"{shorten(ylabel)} val {shorten(eval_metric)}")
        ax.set_ylabel(f"{shorten(ylabel)} val {shorten(eval_metric)}")
    else:
        ax.set_xlabel("Seeds")
        ax.set_ylabel("Seeds")

    # plt.title(save_file.split("_perf_metric=")[0])
    plt.savefig("heat_" + save_file)
    plt.show()


def plot_heatmap(
    heats: np.ndarray,
    order: List[int],
    ticklabels,
    save_file,
    tick_typs="models",
    ylabel="MNLI",
    eval_metric="accuracy",
):
    ordered_heats = np.array([[heats[i][j] for j in order] for i in order])
    heatmap(
        ordered_heats, ticklabels, ticklabels, save_file, tick_typs, ylabel, eval_metric
    )


def lineplot(
    xs,
    ys,
    indices,
    save_file,
    ylabel,
    xlabel=r"$\|\|w-c_1\|\|-\|\|w-c_2\|\|$",
    eval_metric="accuracy",
):
    plt.figure(figsize=(10, 8), dpi=100)
    mpl.rcParams["legend.fontsize"] = 30
    m, b = np.polyfit(xs, ys, 1)

    for k, index in enumerate(indices):
        plt.plot(
            [xs[i] for i in index], [ys[j] for j in index], "o", label=f"cluster-{k}"
        )

    plt.plot(xs, m * xs + b)
    plt.xlabel(xlabel)
    plt.ylabel(f"{shorten(eval_metric)} on {shorten(ylabel)}")
    if len(indices) > 1:
        plt.legend()
    pearson_corr = f"{pearsonr(xs, ys)[0]:.3f}"
    print("Pearson correlation coefficient:", pearson_corr)
    plt.title(r"pearsonr$=$" + rf"${pearson_corr}$")
    plt.savefig("scatter_" + save_file)
    plt.show()


def quiver(
    time_wise_xs,
    time_wise_ys,
    time_wise_indices,
    times,
    save_file,
    ylabel,
    xlabel=r"$\|\|w-c_1\|\|-\|\|w-c_2\|\|$",
    eval_metric="accuracy",
    time_wise_cluster_changes=None,
):
    assert len(time_wise_indices) == len(time_wise_xs) == len(time_wise_ys)

    if time_wise_cluster_changes is None:
        time_wise_cluster_changes = [[False] * len(time_wise_xs[0])] * len(time_wise_xs)

    plt.figure(figsize=(14, 12), dpi=100)

    for n, (xs, ys, indices) in enumerate(
        zip(time_wise_xs, time_wise_ys, time_wise_indices)
    ):
        for k, index in enumerate(indices):
            plt.plot(
                [xs[i] for i in index],
                [ys[j] for j in index],
                "o",
                label=f"cluster-{k}@{times[n]}steps",
            )

    for m in range(1, len(time_wise_xs)):
        is_cluster_changed = time_wise_cluster_changes[m - 1]
        changed_x_prev = [
            elem
            for elem, changed in zip(time_wise_xs[m - 1], is_cluster_changed)
            if changed
        ]
        changed_y_prev = [
            elem
            for elem, changed in zip(time_wise_ys[m - 1], is_cluster_changed)
            if changed
        ]
        changed_x = [
            elem
            for elem, changed in zip(time_wise_xs[m], is_cluster_changed)
            if changed
        ]
        changed_y = [
            elem
            for elem, changed in zip(time_wise_ys[m], is_cluster_changed)
            if changed
        ]

        unchanged_x_prev = [
            elem
            for elem, changed in zip(time_wise_xs[m - 1], is_cluster_changed)
            if not changed
        ]
        unchanged_y_prev = [
            elem
            for elem, changed in zip(time_wise_ys[m - 1], is_cluster_changed)
            if not changed
        ]
        unchanged_x = [
            elem
            for elem, changed in zip(time_wise_xs[m], is_cluster_changed)
            if not changed
        ]
        unchanged_y = [
            elem
            for elem, changed in zip(time_wise_ys[m], is_cluster_changed)
            if not changed
        ]

        changed_Q = plt.quiver(
            changed_x_prev,
            changed_y_prev,
            [e2 - e1 for (e2, e1) in zip(changed_x, changed_x_prev)],
            [e2 - e1 for (e2, e1) in zip(changed_y, changed_y_prev)],
            alpha=0.8,
            width=0.0065,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="tab:pink",
            label="switching clusters",
        )

        # plt.quiverkey(changed_Q, 0.1, 0.2, 2, label="cluster crossing", labelcolor="tab:pink")

        unchanged_Q = plt.quiver(
            unchanged_x_prev,
            unchanged_y_prev,
            [e2 - e1 for (e2, e1) in zip(unchanged_x, unchanged_x_prev)],
            [e2 - e1 for (e2, e1) in zip(unchanged_y, unchanged_y_prev)],
            alpha=0.3,
            width=0.0065,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="tab:gray",
        )

        # plt.quiverkey(unchanged_Q, 0.1, 0.1, 2, label="trapped models", labelcolor="tab:gray")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel(xlabel)
    plt.ylabel(f"{shorten(eval_metric)} on {shorten(ylabel)}")
    plt.title(r"Movement in $\mathfrak{CG}$ metric space with training")
    plt.savefig("quiver_" + save_file)
    plt.show()


def plot_quiver(
    time_wise_heats,
    time_wise_accs,
    time_wise_clusters,
    times,
    save_file,
    perf_metric,
    eval_metric="accuracy",
):
    time_wise_xs = []
    time_wise_ys = []
    time_wise_indices = []
    time_wise_cluster_changes = []

    for j, (heats, clusters, accs) in enumerate(
        zip(time_wise_heats, time_wise_clusters, time_wise_accs)
    ):
        center_wise_dists = get_sc_centroid_dists(
            heats,
        )
        assert (
            len(center_wise_dists) == 2
        ), "Can't plot quiver plot for more than 2 clusters."

        xs = [
            center_wise_dists[0][i] - center_wise_dists[1][i]
            for i in range(len(center_wise_dists[0]))
        ]

        ys = [accs[k] for k in accs]

        assert j == 0 or list(accs.keys()) == acc_keys

        acc_keys = list(accs.keys())

        indices = [[acc_keys.index(k) for k in cluster] for cluster in clusters]
        time_wise_xs.append(xs)
        time_wise_ys.append(ys)
        time_wise_indices.append(indices)
        if j >= 1:
            is_cluster_changed = [
                (
                    (k in time_wise_clusters[j - 1][1])
                    and (
                        k in time_wise_clusters[j][0]
                        or time_wise_xs[-1][n] < time_wise_xs[-2][n]
                    )
                )
                for n, k in enumerate(acc_keys)
            ]

            time_wise_cluster_changes.append(is_cluster_changed)

    with mpl.rc_context({"legend.fontsize": 20}):
        quiver(
            time_wise_xs,
            time_wise_ys,
            time_wise_indices,
            times,
            save_file,
            perf_metric,
            eval_metric=eval_metric,
            time_wise_cluster_changes=time_wise_cluster_changes,
        )


def hist_plot(accs, indices, save_file, perf_metric, eval_metric):
    plt.figure(figsize=(10, 8), dpi=100)

    with mpl.rc_context(
        {
            "axes.titlesize": 50,
            "axes.labelsize": 50,
            "xtick.labelsize": 40,
            "ytick.labelsize": 40,
            "legend.fontsize": 34,
        }
    ):
        acc_keys = list(accs.keys())
        eval_metric = f"{shorten(eval_metric)} on {shorten(perf_metric)}"
        inter_dict = {
            "": [],
            eval_metric: [],
        }

        for i, index in enumerate(indices):
            for e in index:
                inter_dict[""].append(f"cluster-{i}")
                inter_dict[eval_metric].append(accs[acc_keys[e]])

        ax = sns.histplot(
            data=pd.DataFrame(inter_dict),
            hue="",
            x=eval_metric,
            kde=True,
            legend=False,
            bins=12,
        )

        plt.savefig("hist_" + save_file)
        plt.show()


def plot_lineplot(
    heats, accs, clusters, save_file, perf_metric, eval_metric="accuracy"
):
    center_wise_dists = get_sc_centroid_dists(
        heats,
    )
    assert len(center_wise_dists) == 2, "Can't plot lineplot for more than 2 clusters."

    xs = [
        center_wise_dists[0][i] - center_wise_dists[1][i]
        for i in range(len(center_wise_dists[0]))
    ]

    ys = [accs[k] for k in accs]
    acc_keys = list(accs.keys())

    indices = [[acc_keys.index(k) for k in cluster] for cluster in clusters]

    lineplot(
        np.array(xs),
        np.array(ys),
        indices,
        save_file,
        perf_metric,
        eval_metric=eval_metric,
    )
    hist_plot(accs, indices, save_file, perf_metric, eval_metric)


def plot_score_dists(metrics1, metrics2, grp1_name, grp2_name, save_file, sharex=False):
    metrics = {"group": [], "keys": [], "loss_or_acc": []}
    for k in metrics1:
        metrics["keys"] += [k] * (len(metrics1[k]) + len(metrics2[k]))
        metrics["loss_or_acc"] += metrics1[k] + metrics2[k]
        metrics["group"] += [grp1_name] * len(metrics1[k]) + [grp2_name] * len(
            metrics2[k]
        )

    df = pd.DataFrame(metrics)
    g = sns.FacetGrid(
        df,
        hue="group",
        col="keys",
        sharex=sharex,
        margin_titles=True,
        col_wrap=3,
    )
    g.map(sns.histplot, "loss_or_acc", kde=True)
    g.add_legend()
    g.savefig(save_file)
    plt.show()


def annotate(data, **kws):
    n = len(data)
    ax = plt.gca()
    ax.text(0.1, 0.8, f"N = {int(n/10)}", transform=ax.transAxes, fontsize=20)


def plot_plain_valley_peaks(df, save_file, col, x_axis, y_axis, hue_by):
    plt.figure(figsize=(10, 8), dpi=100)

    with mpl.rc_context(
        {
            "lines.linewidth": 2,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 19,
        }
    ):
        n_sample = 8
        grid = sns.FacetGrid(
            df,
            margin_titles=True,
            col=col,
            row=hue_by,
            sharex="all",
            sharey="row",
            # col_wrap=3,
            height=3,
            aspect=1.5,
            hue=hue_by,
        )

        # Draw a line plot to show the trajectory of each random walk
        grid.map(sns.lineplot, x_axis, y_axis, marker="o")
        grid.map_dataframe(annotate)

        # grid.add_legend()
        grid.set(xticks=list(range(10)))
        grid.set_xticklabels(["(1, 0)"] + [""] * n_sample + ["(0, 1)"])

        grid.fig.subplots_adjust(top=0.85)
        # grid.fig.suptitle(save_file[:-4], size=15)
        for ax in grid.axes.flat:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        grid.savefig(save_file)
        plt.show()
