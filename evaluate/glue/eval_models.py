import argparse

import torch
from transformers import AutoTokenizer

import constellations.utils.eval_utils as util

from constellations.model_loaders.modelling_utils import get_logits_converter, get_pred_fn
from constellations.model_loaders.load_model import get_models_ft_with_prefix, get_sequence_classification_model
from constellations.dataloaders.loader import get_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
def main(args):
    is_feather_bert = "feather" in args.model_name

    if is_feather_bert:
        mnli_label_dict = {"contradiction": 0, "entailment": 1, "neutral": 2}
    else:        
        mnli_label_dict = {"contradiction": 2, "entailment": 0, "neutral": 1}
    
    hans_label_dict = {"entailment": 0, "non-entailment": 1}
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    if args.dataset=="hans" and args.all_data:
        input_target_loader = get_loader(args, tokenizer, mnli_label_dict, 
                                         heuristic_wise=["lexical_overlap",
                                                         "constituent", "subsequence"],
                                         onlyNonEntailing=False)
    else:
        input_target_loader = get_loader(args, tokenizer, mnli_label_dict)
    
    mnli_logits_to_hans = get_logits_converter(mnli_label_dict, hans_label_dict)
    
    logit_converter = mnli_logits_to_hans if args.dataset=="hans" else None
    
    pred_fn   = get_pred_fn(pred_type="prob", logit_converter_fn = logit_converter)
    
    local_model_dir = (None if args.local_dir_prefix is None
                       else args.local_dir_prefix + args.model_name[len(args.base_models_prefix):])
    
    model = get_sequence_classification_model(args.model_name, num_steps=args.num_steps,
                                              from_flax=not is_feather_bert,
                                              local_dir=local_model_dir).to(device)

    if args.dataset=="cola" or args.dataset=="qqp":
        label_dict = {elem:i for i, elem in 
                      enumerate(input_target_loader.hf_loader.dataset.info.features["label"].names)}
    elif args.dataset=="paws":
        label_dict = {"not_duplicate":0, "duplicate": 1}
    else:
        label_dict = mnli_label_dict if args.dataset=="mnli" else hans_label_dict
    
    util.store_outs(input_target_loader, args.write_file, tokenizer, model, pred_fn, label_dict)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluate models on HF-Hub.")

    parser.add_argument(
        "--base_models_prefix",
        type=str,
        help="Common prefix of models to be loaded(e.g. 'connectivity/bert_ft_qqp-')",
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="bert-base-uncased",
        help="Model name to be used to load the tokenizer \
              for the models, if tokenizer not found in\
              args.base_models_prefix+args.models[i]. (default: bert-base-uncased)",
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
        choices=["train", "test", "in_domain_dev", "out_domain_dev", "validation", "dev_and_test"],
        help="data split to use: [train, test](for mnli/hans) or \
            [train, in_domain_dev, out_domain_dev, validation](for cola) or\
            [dev_and_test, train](for paws) or [train, validation, test](qqp).\
            For MNLI, 'test' resolves to 'validation_matched', and for HANS,\
            'test' resolves to 'validation'. (default: test)",
    )
    
    parser.add_argument(
        "--all_data",
        action="store_true",
        default=False,
        help="whether to use all data for evaluation on HANS dataset. By default, \
            only nonEntailing samples of lexical_overlap heuristic are used.",
    )
    
    parser.add_argument(
        "--write_file_prefix",
        type=str,
        required=True,
        help="Prefix of filename where results are to be written.",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="The suffixes to add to base_models_prefix to get the models\
              to evaluate.")
    
    parser.add_argument(
        "--num_steps",
        type=int,
        help="Load model on the remote at these number of steps, \
            for evaluation. A commit with this number of '\d+ steps' in its\
            commit message, must be present on the remote.By default, latest\
            model will be loaded."
    )

    parser.add_argument(
        "--paws_data_dir",
        type=str,
        default="../../paws_final",
        help="Data directory having final paws-qqp files(default: ../../paws_final)."
    )
    
    parser.add_argument(
        "--local_dir_prefix",
        type=str,
        help="Local directory prefix to fetch commits from(if repository is\
            already cloned on local machine). For use with num_steps.\
            If not specified commit info will be fetched\
            into dummy directory from huggingface.co/",
    )

    args = parser.parse_args()
    
    if args.models is None:
        suffixes = [elem[len(args.base_models_prefix):] for elem in 
                        get_models_ft_with_prefix(args.base_models_prefix)]

        print(f"Evaluating models: ", 
              [args.base_models_prefix+suffix for suffix in suffixes], 
              flush=True)
    else:
        suffixes = args.models
    
    for suffix in suffixes:
        args.model_name = args.base_models_prefix+suffix
        print(f"Evaluating model {args.model_name}", flush=True)
        args.write_file = args.write_file_prefix+suffix+".json"    
        main(args)
