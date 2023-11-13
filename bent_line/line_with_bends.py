import argparse
import tabulate

import os
import sys
import glob

import torch
import torch.nn as nn
import evaluate
from transformers import AutoTokenizer

import time

from constellations.dataloaders.loader import get_train_test_loaders
from constellations.simplexes.models.orig_basic_simplex import SimplicialComplex
from constellations.simplexes.orig_utils import train_transformer_epoch, eval_model
from constellations.model_loaders.load_model import get_sequence_classification_model
from constellations.model_loaders.modelling_utils import get_pred_fn, get_criterion_fn, get_logits_converter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    first_model_name = args.base_models_prefix.strip('/').split('/')[-1][:-1]

    save_dir = first_model_name + "_lr_" + str(args.lr_init) + "bent_complex"
    save_dir = os.path.join(save_dir, 'indices_'+args.indices.replace(',', '_'))
    if args.num_steps is not None:
        save_dir = os.path.join(save_dir,'num_steps_'+args.num_steps.replace(',', '_'))

    print('Preparing directory %s' % save_dir)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'base_command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    save_file_prefix = os.path.join(save_dir.rstrip('/'), 'vertex_')
    if glob.glob(save_file_prefix+'*'):
        print(f"Found pre-existing checkpoint with prefix {save_file_prefix}. Loading it.")
        args.load_pretrained = save_file_prefix
    elif args.begin_from_dataloader!=0:
        raise FileNotFoundError(f'Unexpected. Trying to start from {args.begin_from_dataloader}-th'
                                f' dataloader. But no checkpoint corres. to training on previous'
                                f' epochs found using prefix: {save_file_prefix}')

    mnli_label_dict = {"contradiction": 0, "entailment": 1, "neutral": 2}
    hans_label_dict = {"entailment": 0, "non-entailment": 1}

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    trainloaders, testloader = get_train_test_loaders(args, tokenizer, mnli_label_dict,
                                                      heuristic_wise=["lexical_overlap",
                                                      "constituent", "subsequence"],
                                                      onlyNonEntailing=False)
    ce_loss = nn.CrossEntropyLoss()

    mnli_logits_to_hans = get_logits_converter(mnli_label_dict, hans_label_dict)

    logit_converter = mnli_logits_to_hans if args.dataset=="hans" else None

    criterion = get_criterion_fn(ce_loss, logit_converter)
    pred_fn   = get_pred_fn(logit_converter_fn = logit_converter)

    if args.dataset in ["mnli", "hans"]:
        metric = evaluate.load("accuracy", experiment_id=args.experiment_id)
    elif args.dataset in ["qqp"]:
        metric = evaluate.load("glue", "qqp", experiment_id=args.experiment_id)

    model_ids = args.indices.split(',')
    if args.num_steps is not None:
        num_steps = [int(elem.strip()) for elem in args.num_steps.split(',')]
    else:
        num_steps = None

    is_fthr = 'feather' in args.base_models_prefix
    if args.load_pretrained is not None:
        model_path = args.base_models_prefix + ("0" if is_fthr and int(model_ids[0])<10 else "") + model_ids[0]
        model = get_sequence_classification_model(model_path)
        print(f'Loading Pretrained models from {args.load_pretrained}')
        simplicial_complex = SimplicialComplex.from_pretrained(model, args.load_pretrained,
                                                               fix_points=[True, True,]+[False,]*args.n_bends)
    else:
        #Load End Point Models
        for ii, id in enumerate(model_ids):
            model_path = args.base_models_prefix + ("0" if is_fthr and int(id)<10 else "") + id
            if num_steps is not None:
                model = get_sequence_classification_model(model_path, num_steps=num_steps[ii])
            else:
                model = get_sequence_classification_model(model_path)

            if ii == 0:
                simplicial_complex = SimplicialComplex(model, num_vertices=1,
                                                       fixed_points=[True])
            else:
                simplicial_complex.add_base_vertex(model)

            del model

        simplicial_complex = simplicial_complex.to(device)

        #Create Connecting Points.
        prev_vertex = 0
        for ii in range(args.n_bends):
            linking = [0]*simplicial_complex.num_vertices
            linking[prev_vertex] = 1
            if ii==args.n_bends-1:
                linking[1] = 1
            coeffs = [(ii+1)/(args.n_bends+1), (args.n_bends-ii)/(args.n_bends+1)]+[0]*(simplicial_complex.num_vertices-2)
            simplicial_complex.add_conn_vertex(linking=linking, coeffs=coeffs)
            prev_vertex = simplicial_complex.num_vertices-1

    simplicial_complex = simplicial_complex.to(device)

    optimizer = torch.optim.Adam(
                    simplicial_complex.parameters(),
                    lr=args.lr_init,
                    weight_decay=args.wd
                )

    columns = ['vert', 'ep', 'lr', 'tr_loss',
                'tr_acc', 'te_loss', 'te_acc', 'time', 'volume']

    total_sub_epochs = 0

    ## add connecting points and train ##

    for epoch in range(args.epochs):
        if total_sub_epochs==args.total_sub_epochs:
            break
        for sub_epoch in range(args.break_epochs):

            if total_sub_epochs==args.total_sub_epochs:
                break

            time_ep = time.time()

            train_res = train_transformer_epoch(
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
                test_res = eval_model(testloader, simplicial_complex,
                                      criterion, pred_fn, metric)
            else:
                test_res = {'loss': None, 'accuracy': None}

            time_ep = time.time() - time_ep

            lr = optimizer.param_groups[0]['lr']

            values = [simplicial_complex.num_vertices, total_sub_epochs + 1, lr,
                      train_res['loss'], train_res['accuracy'],
                      test_res['loss'], test_res['accuracy'], time_ep,
                      simplicial_complex.total_volume().item()]

            table = tabulate.tabulate([values], columns,
                                      tablefmt='simple',)
            if total_sub_epochs % 40 == 0:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
            else:
                table = table.split('\n')[2]

            total_sub_epochs += 1
            print(table, flush=True)

            simplicial_complex.save_params(save_file_prefix)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Line with n-bends")

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
        "--num_steps",
        type=str,
        required=False,
        help="Comma separated list of step numbers(integers) from where to\
              load the models."
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
        "--n_bends",
        type=int,
        default=2,
        help="number of connecting vertices for the complex.",
    )

    parser.add_argument(
        "--lr_init",
        type=float,
        default=1e-5,
        help="initial learning rate (default: 2e-5)",
    )

    parser.add_argument(
        "--wd",
        type=float,
        default=0.,
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
        choices=["mnli", "hans", "qqp"],
        help="dataset [mnli or hans or qqp]",
    )

    parser.add_argument(
        "--break_epochs",
        type=int,
        default=1,
        help="Number of parts to break each epoch into",
    )

    parser.add_argument(
        "--experiment_id",
        type=str,
        default='',
        required=False,
        help="Experiment ID, required if multiple experiments are sharing same file system.",
    )

    parser.add_argument(
        "--total_sub_epochs",
        type=int,
        default=-1,
        help="Total number of sub epochs to train for.",
    )

    parser.add_argument(
        "--load_pretrained",
        type=str,
        required=False,
        help="Path to load models of a SimplicialComplex saved halfway through training."
    )

    parser.add_argument(
        "--begin_from_dataloader",
        type=int,
        default=0,
        required=False,
        help=("If specified, the training will being from the n-th train dataloader."
              "Where n is the number specified.")
    )

    args = parser.parse_args()

    if args.total_sub_epochs == -1:
        args.total_sub_epochs = args.break_epochs * args.epochs

    if args.total_sub_epochs>args.break_epochs*args.epochs:
        raise ValueError("Total sub epochs cannot be greater than epochs * break_epochs")

    if args.begin_from_dataloader>args.break_epochs:
        raise ValueError(f"The dataloader to begin with ({args.begin_from_dataloader})"
                         f" should be <= number of sub-parts within an epoch ({args.break_epochs})")

    print("Dataloaders index offset added:", args.begin_from_dataloader)

    main(args)
