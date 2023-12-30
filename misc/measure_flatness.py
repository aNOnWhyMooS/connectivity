import argparse

import torch
import torch.nn as nn

import constellations.legacy.common_utils as cu

import constellations.simplexes.orig_utils as simp_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Measure the ɛ-sharpness of a BERT model."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the model to use."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="input batch size (default: 128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=8e-5,
        help="initial learning rate (default: 8e-5)",
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="sgd",
        choices=["sgd", "adam"],
        help="optimizer to use (default: sgd)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="number of gradient accumulation steps",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnli",
        choices=["mnli", "hans"],
        help="dataset [mnli or hans]",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=10,
        help="number of batches to use for evaluation and training.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-5,
        help="epsilon for the epsilon-sharpness metric",
    )
    parser.add_argument(
        "--num_random_dirs",
        type=int,
        required=False,
        help="If provided, the loss will be maximized in only these many directions.\
                                                                             By default, the loss is maximized in all directions.",
    )
    parser.add_argument(
        "--heuristics",
        type=str,
        nargs="+",
        default=["lexical_overlap", "constituent", "subsequence"],
        help="heuristics to use for HANS dataset",
    )
    parser.add_argument(
        "--only_non_entailing",
        action="store_true",
        help="only use non-entailing examples for HANS",
    )
    return parser


def main(args):
    mnli_label_dict = {"contradiction": 0, "entailment": 1, "neutral": 2}
    hans_label_dict = {"entailment": 0, "non-entailment": 1}

    input_target_loader = cu.get_loader(
        args,
        mnli_label_dict,
        heuristic_wise=args.heuristics,
        onlyNonEntailing=args.only_non_entailing,
    )

    ce_loss = nn.CrossEntropyLoss()

    mnli_logits_to_hans = cu.get_logits_converter(mnli_label_dict, hans_label_dict)

    logit_converter = mnli_logits_to_hans if args.dataset == "hans" else None

    criterion = cu.get_criterion_fn(ce_loss, logit_converter)

    model = cu.get_sequence_classification_model(args.model, args.base_model).to(device)

    if args.opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr_init,
        )

    flatness = simp_utils.flatness_measure(
        input_target_loader,
        model,
        criterion,
        optimizer,
        args.gradient_accumulation_steps,
        args.epsilon,
        num_random_dirs=args.num_random_dirs,
    )

    print(f"ɛ-sharpness of {args.model} is {flatness}", flush=True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.num_exs = args.batch_size * args.n_batches
    args.split = "test" if args.dataset == "mnli" else "train"
    args.base_model = "bert-base-uncased"
    main(args)
