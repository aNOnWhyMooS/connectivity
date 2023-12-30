import argparse
import tabulate

import os
import sys
import time

import torch
import torch.nn as nn
from datasets import load_metric

import constellations.simplexes.orig_utils as simp_utils

import constellations.legacy.common_utils as cu

from constellations.simplexes.models.orig_basic_simplex import SimplicialComplex

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    first_model_name = args.base_models_prefix.strip("/").split("/")[-1]
    savedir = "./saved-outputs/model" + first_model_name + "/"
    print("Preparing directory %s" % savedir)
    os.makedirs(savedir, exist_ok=True)
    with open(os.path.join(savedir, "base_command.sh"), "w") as f:
        f.write(" ".join(sys.argv))
        f.write("\n")

    mnli_label_dict = {"contradiction": 0, "entailment": 1, "neutral": 2}
    hans_label_dict = {"entailment": 0, "non-entailment": 1}

    trainloaders, testloader = cu.get_train_test_loaders(
        args,
        mnli_label_dict,
        heuristic_wise=["lexical_overlap", "constituent", "subsequence"],
        onlyNonEntailing=False,
    )
    ce_loss = nn.CrossEntropyLoss()

    mnli_logits_to_hans = cu.get_logits_converter(mnli_label_dict, hans_label_dict)

    logit_converter = mnli_logits_to_hans if args.dataset == "hans" else None

    criterion = cu.get_criterion_fn(ce_loss, logit_converter)
    pred_fn = cu.get_pred_fn(logit_converter_fn=logit_converter)

    metric = load_metric("accuracy")

    model_ids = args.indices.split(",")
    # all_base_models = []
    for ii, id in enumerate(model_ids):
        model_path = args.base_models_prefix + id
        model = cu.get_sequence_classification_model(model_path)
        # all_base_models.append(model)
        if ii == 0:
            simplicial_complex = SimplicialComplex(
                model, num_vertices=1, fixed_points=[True]
            )
        else:
            simplicial_complex.add_base_vertex(model)

        del model

    simplicial_complex = simplicial_complex.to(device)

    columns = [
        "vert",
        "ep",
        "lr",
        "tr_loss",
        "tr_acc",
        "te_loss",
        "te_acc",
        "time",
        "volume",
    ]
    num_base_vertices = simplicial_complex.num_vertices
    total_sub_epochs = 0

    ## add connecting points and train ##

    for epoch in range(args.epochs):
        for sub_epoch in range(args.break_epochs):
            if total_sub_epochs % args.sub_epochs_per_vertex == 0:
                if simplicial_complex.num_vertices == num_base_vertices + args.n_conn:
                    break
                simplicial_complex.add_conn_vertex()
                # Current keeping ```linking```` as default ^
                # i.e., all conn verts to be connected to all base vertices
                simplicial_complex = simplicial_complex.to(device)
                optimizer = torch.optim.Adam(
                    simplicial_complex.parameters(),
                    lr=args.lr_init,
                    weight_decay=args.wd,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.sub_epochs_per_vertex
                )

            time_ep = time.time()
            train_res = simp_utils.train_transformer_epoch(
                trainloaders[sub_epoch],
                simplicial_complex,
                criterion,
                optimizer,
                args.n_sample,
                vol_reg=1e-4,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                pred_fn=pred_fn,
                metric=metric,
            )

            eval_ep = total_sub_epochs % args.eval_freq == args.eval_freq - 1

            if eval_ep:
                test_res = simp_utils.eval(
                    testloader, simplicial_complex, criterion, pred_fn, metric
                )
            else:
                test_res = {"loss": None, "accuracy": None}

            time_ep = time.time() - time_ep

            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            values = [
                simplicial_complex.num_vertices,
                total_sub_epochs + 1,
                lr,
                train_res["loss"],
                train_res["accuracy"],
                test_res["loss"],
                test_res["accuracy"],
                time_ep,
                simplicial_complex.total_volume().item(),
            ]

            table = tabulate.tabulate(
                [values],
                columns,
                tablefmt="simple",
            )
            if total_sub_epochs % 40 == 0:
                table = table.split("\n")
                table = "\n".join([table[1]] + table)
            else:
                table = table.split("\n")[2]

            total_sub_epochs += 1
            print(table, flush=True)

    save_file_prefix = (
        first_model_name + "_lr_" + str(args.lr_init) + "complex_simplex_v"
    )
    simplicial_complex.save_params(save_file_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simplicial complex")

    parser.add_argument(
        "--base_models_prefix",
        type=str,
        required=True,
        help="Common prefix of models to be loaded",
    )

    parser.add_argument(
        "--indices",
        type=str,
        required=True,
        help="Indices of models to be loaded seperated by commas",
    )

    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Model name to be used to load the tokenizer \
             for the models.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="input batch size (default: 128)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs to train each vertex in the simplex",
    )

    parser.add_argument(
        "--n_conn",
        type=int,
        default=2,
        help="number of connecting vertices for the complex.",
    )

    parser.add_argument(
        "--lr_init",
        type=float,
        default=8e-5,
        help="initial learning rate (default: 0.03)",
    )

    parser.add_argument(
        "--wd",
        type=float,
        default=0.0,
        help="weight decay",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=15,
        help="number of gradient accumulation steps",
    )

    parser.add_argument(
        "--n_sample",
        type=int,
        default=5,
        help="number of models to sample from the simplex to calculate \
              loss per iteration",
    )

    parser.add_argument(
        "--eval_freq",
        type=int,
        default=1,
        help="evaluate every n epochs",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnli",
        choices=["mnli", "hans"],
        help="dataset [mnli or hans]",
    )

    parser.add_argument(
        "--break_epochs",
        type=int,
        default=1,
        help="Number of parts to break each epoch into",
    )

    parser.add_argument(
        "--sub_epochs_per_vertex",
        type=int,
        default=1,
        help="Number of sub-epochs after which to add new vertex",
    )

    args = parser.parse_args()

    if args.break_epochs * args.epochs < args.sub_epochs_per_vertex:
        raise ValueError(
            f"Total number of sub-epochs {args.break_epochs*args.epochs} must be\
                            greater than or equal to number of sub-epochs per vertex: \
                            {args.sub_epochs_per_vertex}."
        )

    main(args)
