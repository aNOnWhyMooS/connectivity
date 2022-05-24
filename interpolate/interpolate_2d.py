from typing import OrderedDict
import copy
import pickle
import argparse
import collections

import numpy as np
import torch
import torch.nn as nn

from constellations.legacy.get_datasets import load_metric

import constellations.simplexes.surfaces as surfaces
import constellations.simplexes.orig_utils as util
from constellations.simplexes.orig_surfaces import get_sd_basis

import constellations.legacy.common_utils as cu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def perturb_pars(w1: OrderedDict[str, torch.Tensor], 
                 w2: OrderedDict[str, torch.Tensor], 
                 coeff1: float, coeff2: float,
                 model: nn.Module) -> None:
    """Linearly combines weights w1 and w2 as coeff1*w1 and coeff2*w2 and perturbs 
    parameters of provided model, in this direction.
    Args:
        w1:     State dict of first basis vector(no buffers).
        w2:     State dict of second basis vector(no buffers).
        coeff1: Coefficient for scaling weights in w1.
        coeff2: Coefficient for scaling weights in w2.
        model:   The model whose weights to perturb by the linear combination of w1 and w2.
    """
    linear_comb = collections.OrderedDict()
    for (k1, v1), (k2, v2) in zip(w1.items(), w2.items()):
        if k1!=k2:
            raise ValueError(f"Mis-matched keys {k1} and {k2} encountered while \
                               forming linear combination of weights.")
        linear_comb[k1] = coeff1*v1+coeff2*v2
    
    new_state_dict = model.state_dict()
    for k, v in linear_comb.items():
        new_state_dict[k] += v.to(new_state_dict[k].device)
    
    model.load_state_dict(new_state_dict)
    model.to(device)

def main(args):
    mnli_label_dict = {"contradiction": 0, "entailment": 1, "neutral": 2}
    hans_label_dict = {"entailment": 0, "non-entailment": 1}

    if args.dataset=="hans" and args.all_data:
        input_target_loader = cu.get_loader(args, mnli_label_dict, 
                                            heuristic_wise=["lexical_overlap", 
                                                            "constituent", "subsequence"],
                                            onlyNonEntailing=False)
    else:
        input_target_loader = cu.get_loader(args, mnli_label_dict)
    
    ce_loss = nn.CrossEntropyLoss()
    
    mnli_logits_to_hans = cu.get_logits_converter(mnli_label_dict, hans_label_dict)
            
    logit_converter = mnli_logits_to_hans if args.dataset=="hans" else None
    
    criterion = cu.get_criterion_fn(ce_loss, logit_converter)
    pred_fn   = cu.get_pred_fn(pred_type="prob" if args.metric=="ECE" else "argmax",
                              logit_converter_fn = logit_converter)

    if args.metric=="accuracy":
        metric = load_metric("accuracy")
    elif args.metric=="ECE":
        metric = load_metric("ECE", n_bins=10)
    
    w1 = cu.get_sequence_classification_model(
        args.base_models_prefix + str(args.base1)
    )
    
    w2 = cu.get_sequence_classification_model(
        args.base_models_prefix + str(args.base2)
    )
    
    anchor = cu.get_sequence_classification_model(
        args.base_models_prefix + str(args.anchor)
    )
    
    basis1, basis2, removed_comp_len, swapped_basis, perp_scaling_factor = get_sd_basis(anchor, w1, w2)
    del w1, w2

    locations = {"w0" : (0,0), 
                 "w1" : (1,0) if not swapped_basis else (removed_comp_len, perp_scaling_factor),
                 "w2" : (1,0) if swapped_basis else (removed_comp_len, perp_scaling_factor) }
    
    print("Basis-1 model location:", locations["w1"])
    print("Basis-2 model location:", locations["w2"])

    n_pts = args.n_pts_per_unit*(args.range - (-args.range))
    loss_surf = torch.zeros(n_pts, n_pts)
    acc_surf = torch.zeros(n_pts, n_pts)
    ece_surf = torch.zeros(n_pts, n_pts)
    vec = np.linspace(-args.range, args.range, n_pts)
    coef_samples = vec.tolist()
    
    original_state = copy.deepcopy(anchor.state_dict())
    
    for i, coeff1 in enumerate(coef_samples):
        for j, coeff2 in enumerate(coef_samples):
            perturb_pars(basis1, basis2, coeff1, coeff2, anchor)
            metrics = util.eval(input_target_loader, anchor, 
                                criterion, pred_fn, metric)
            loss_surf[i, j] = metrics["loss"]
            acc_surf[i, j] =  metrics["accuracy"]
            
            print(f"Metrics for coefficients ({coeff1}, {coeff2}) \
                    for the given bases:", metrics)
            
            if args.metric=="ECE":
                ece_surf[i, j] = metrics["ECE"]
            
            anchor.load_state_dict(original_state)
    
    X, Y = np.meshgrid(vec, vec)
    loss_surface = (X, perp_scaling_factor*Y, loss_surf)
    acc_surface = (X, perp_scaling_factor*Y, acc_surf)
    if args.metric=="ECE":
        ece_surface = (X, perp_scaling_factor*Y, ece_surf)
    
    print('Surfaces computed...')
    all_data = "AllData_" if (args.all_data and args.dataset=="hans") else ""
    save_name = 'saved-outputs/'+ all_data + f"{args.dataset}_{args.split}_{args.anchor}_{args.base1}_{args.base2}_{args.range}"
    pickle.dump((loss_surface, locations), open(save_name + '_loss_surface.pkl', 'wb'))
    surfaces.plot_loss_surface(loss_surface, save_name + '_loss_surface_2d', three_d=False, locations=locations)
    surfaces.plot_loss_surface(loss_surface, save_name + '_loss_surface_3d')
    pickle.dump((acc_surface, locations), open(save_name + '_acc_surface.pkl', 'wb'))
    surfaces.plot_loss_surface(acc_surface, save_name + '_acc_surface_2d', three_d=False, locations=locations)
    surfaces.plot_loss_surface(acc_surface, save_name + '_acc_surface_3d')
    
    if args.metric=="ECE":
        pickle.dump((ece_surface, locations), open(save_name + '_ece_surface.pkl', 'wb'))
        surfaces.plot_loss_surface(ece_surface, save_name + '_ece_surface_2d', three_d=False, locations=locations)
        surfaces.plot_loss_surface(ece_surface, save_name + '_ece_surface_3d')
    
    print('Surfaces saved')

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="2-D interpolations on a plane made of 3 models.")

    parser.add_argument(
        "--base_models_prefix",
        type=str,
        required=True,
        help="Common prefix of models to be loaded(e.g. 'connectivity/feather_berts_')",
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
        "--anchor",
        type=str,
        required=True,
        help="Anchor vertex for constructing basis of the plane\
              (this model will be at the origin).",
    )

    parser.add_argument(
        "--base1",
        type=int,
        required=True,
        help="Base-1 model for getting basis\
              (This model will be marked by w0)",
    )

    parser.add_argument(
        "--base2",
        type=int,
        required=True,
        help="Base-1 model for getting basis\
              (This model will be marked by w1)",
    )

    parser.add_argument(
        "--range",
        type=int,
        default=4,
        help="Range of the loss surface. One unit range will reach from anchor\
              to both base1 and base2 models. (Default: 4)",
    )

    parser.add_argument(
        "--n_pts_per_unit",
        type=int,
        default=20,
        help="Number of points to sample from a unit length of an axis.\
              In total, (2*n_pts*range)^2 points are sampled. (default: 20)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="input batch size (default: 128)",
    )    

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnli",
        choices=["mnli", "hans"],
        help="dataset [mnli, hans] (default: mnli)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="data split [train, test]. For MNLI, 'test' resolves to \
            'validation_matched', and for HANS, 'test' resolves to \
            'validation'. (default: test)",
    )

    parser.add_argument(
        "--num_exs",
        type=int,
        default=256,
        help="number of examples used to evaluate each model on \
              the linear interpolation curve. (default: 256)",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="acc",
        choices=["acc", "ECE"],
        help="metric to be used for evaluation, use acc or ECE.\
            With ECE, both acc and ECE surfaces will be shown. (Default: acc)",
    )

    parser.add_argument(
        "--all_data",
        action="store_true",
        default=False,
        help="whether to use all data for evaluation on HANS dataset. By default, \
            only nonEntailing samples of lexical_overlap heuristic are used.",
    )

    args = parser.parse_args()

    main(args)
