import os
import glob
import argparse

import torch
from transformers import AutoTokenizer
from huggingface_hub import HfApi

import constellations.utils.eval_utils as util

from constellations.model_loaders.modelling_utils import (
    get_logits_converter,
    get_pred_fn,
)
from constellations.model_loaders.load_model import get_sequence_classification_model
from constellations.dataloaders.loader import get_loader
from constellations.hf_api_utils import get_all_models, get_all_steps, get_model_type


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_all_model_step_pairs(args):
    ms_pairs = []
    if not HfApi().repo_exists(args.model):
        if args.step == "all":
            models = get_all_models(args.model)
            for model in models:
                steps = get_all_steps(model)
                ms_pairs += [(model, step) for step in steps]
        else:
            models = get_all_models(args.model, args.step)
            ms_pairs = [(model, args.step) for model in models]
    elif args.step == "all":
        steps = get_all_steps(args.model)
        ms_pairs = [(args.model, step) for step in steps]
    else:
        ms_pairs = [(args.model, args.step)]

    return [(*p, i) for p in ms_pairs for i in range(args.n_parts)]


def main(args):
    is_feather_bert = "feather" in args.model

    if is_feather_bert:
        mnli_label_dict = {"contradiction": 0, "entailment": 1, "neutral": 2}
    else:
        mnli_label_dict = {"contradiction": 2, "entailment": 0, "neutral": 1}

    hans_label_dict = {"entailment": 0, "non-entailment": 1}

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, model_max_length=128)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, model_max_length=128)

    if args.dataset == "hans" and args.all_data:
        input_target_loader = get_loader(
            args,
            tokenizer,
            mnli_label_dict,
            heuristic_wise=["lexical_overlap", "constituent", "subsequence"],
            onlyNonEntailing=False,
            n_parts=args.n_parts,
            index=args.index,
        )
    else:
        input_target_loader = get_loader(
            args, tokenizer, mnli_label_dict, n_parts=args.n_parts, index=args.index
        )

    mnli_logits_to_hans = get_logits_converter(mnli_label_dict, hans_label_dict)

    logit_converter = mnli_logits_to_hans if args.dataset == "hans" else None

    pred_fn = get_pred_fn(pred_type="prob", logit_converter_fn=logit_converter)

    model = get_sequence_classification_model(
        path_or_name=args.model,
        num_steps=args.step,
        from_flax=(args.from_model_type == "flax"),
    ).to(device)

    if args.dataset == "cola" or args.dataset == "qqp":
        label_dict = {
            elem: i
            for i, elem in enumerate(
                input_target_loader.hf_loader.dataset.info.features["label"].names
            )
        }
    elif args.dataset == "paws":
        label_dict = {"not_duplicate": 0, "duplicate": 1}
    else:
        label_dict = mnli_label_dict if args.dataset == "mnli" else hans_label_dict

    util.store_outs(
        input_target_loader, args.save_file, tokenizer, model, pred_fn, label_dict
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on HF-Hub.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of model to evaluate as on hf-hub."
        "Or a substring. All models on hf-hub having this substring will "
        "be evaluated.",
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="bert-base-uncased",
        help="Model name to be used to load the tokenizer "
        "(default: bert-base-uncased). Will be only used if"
        "tokenizer can't be loaded from args.model.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="input batch size (default: 512)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnli",
        choices=["mnli", "hans", "cola", "qqp", "paws"],
        help="dataset to evaluate on [mnli, hans, cola, qqp, paws]. (default: mnli)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=[
            "train",
            "test",
            "in_domain_dev",
            "out_domain_dev",
            "validation",
            "dev_and_test",
        ],
        help="data split to use: [train, test](for mnli/hans) or "
        "[train, in_domain_dev, out_domain_dev, validation](for cola) or "
        "[dev_and_test, train](for paws) or [train, validation, test](qqp). "
        "For MNLI, 'test' resolves to 'validation_matched', and for HANS, "
        "'test' resolves to 'validation'. (default: test)",
    )

    parser.add_argument(
        "--all_data",
        action="store_true",
        default=False,
        help="whether to use all data for evaluation on HANS dataset. By default, \
            only nonEntailing samples of lexical_overlap heuristic are used.",
    )

    parser.add_argument(
        "--save_file",
        type=str,
        required=True,
        help="Name of file where to save the evaluated data," "using pickle.",
    )

    parser.add_argument(
        "--step",
        type=str,
        help="The step number at which to fetch the model"
        " specified in args.model."
        "A commit with this number: ' \d+ steps' in its "
        "commit message, must be present on the remote. By default, latest "
        "model will be loaded. Specify step==all to evaluate at all steps.",
    )

    parser.add_argument(
        "--job_id",
        required=False,
        type=int,
        help="In case args.model is a substring, or args.step==all this tells "
        "which exact model to evaluate in the current job.",
    )

    parser.add_argument(
        "--n_parts",
        required=False,
        type=int,
        default=1,
        help="If specified, dataset will be split into these many parts and "
        "each part will be evaluated in separate job.",
    )

    args = parser.parse_args()

    args.paws_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "paws_final",
    )

    if args.job_id is not None:
        args.model, args.step, args.index = get_all_model_step_pairs(args)[args.job_id]
        suffix = args.save_file.split(".")[-1]
        prefix = (
            ".".join(args.save_file.split(".")[:-1])
            + f"_{args.dataset}_{args.split}"
            + f'_{args.model.split("/")[-1]}@{args.step}'
            + f"_{args.index}of{args.n_parts}"
        )

        already_completed = glob.glob(f"{prefix}_*.{suffix}")

        if already_completed:
            print(
                f"Job already completed with logs at:", already_completed, ". Exitting."
            )
            exit(0)

        args.save_file = f"{prefix}_{args.job_id}.{suffix}"
    else:
        ms_pairs = get_all_model_step_pairs(args)
        assert len(ms_pairs) == 1
        args.model, args.step, args.index = ms_pairs[0]

    print(
        f"Evaluating model: {args.model}@{args.step} steps on data: ({args.dataset}, {args.split})"
    )
    args.from_model_type = get_model_type(args.model)
    main(args)
